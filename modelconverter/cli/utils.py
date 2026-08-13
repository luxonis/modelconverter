import shutil
from contextlib import suppress
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from luxonis_ml.nn_archive import is_nn_archive
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import PreprocessingBlock
from luxonis_ml.typing import Params
from requests.exceptions import HTTPError

from modelconverter.utils import (
    ModelconverterException,
    process_nn_archive,
    resolve_path,
    sanitize_net_name,
)
from modelconverter.utils.config import Config
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    CONFIGS_DIR,
    CONVERSION_MARKER,
    MISC_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    in_docker,
)
from modelconverter.utils.filesystem_utils import set_input_base
from modelconverter.utils.hub_requests import Request
from modelconverter.utils.types import DataType, Encoding, Target


class ModelType(str, Enum):
    ONNX = "ONNX"
    IR = "IR"
    PYTORCH = "PYTORCH"
    TFLITE = "TFLITE"
    RVC2 = "RVC2"
    RVC3 = "RVC3"
    RVC4 = "RVC4"
    HAILO = "HAILO"

    @classmethod
    def from_suffix(cls, suffix: str) -> "ModelType":
        if suffix == ".onnx":
            return cls.ONNX
        if suffix == ".tflite":
            return cls.TFLITE
        if suffix in [".xml", ".bin"]:
            return cls.IR
        if suffix in [".pt", ".pth"]:
            return cls.PYTORCH
        raise ValueError(f"Unsupported model format: {suffix}")


def resolve_output_dir(output_dir: str) -> Path:
    """Resolve a user-provided ``--output-dir`` under `OUTPUTS_DIR`.

    Only the host's ``./output`` is mounted into the container, so an
    absolute path -- which ``pathlib`` would let win over the mount point
    -- or one escaping upwards through ``..`` would be written to a
    container-only location and lost when the container is removed. A
    native run writes straight to the host filesystem, where every path
    the user names is reachable, so any path is honored there.
    """
    relative = Path(output_dir)
    if in_docker() and (
        not relative.parts or relative.is_absolute() or ".." in relative.parts
    ):
        raise ModelconverterException(
            f"Invalid `--output-dir` '{output_dir}': it must name a "
            f"directory inside '{OUTPUTS_DIR}'. Pass a relative path "
            "without `..`."
        )
    return OUTPUTS_DIR / relative


def get_output_dir_name(
    target: Target, name: str, output_dir: str | None
) -> Path:
    name = sanitize_net_name(name)
    date = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    if output_dir is not None:
        dest = resolve_output_dir(output_dir)
        if dest.exists():
            # `OUTPUTS_DIR` is the user's `./output`, so a rerun may only clear
            # a directory a previous conversion produced. Inference results
            # carry a marker of their own and are not ours to delete either.
            if not dest.is_dir():
                raise ModelconverterException(
                    f"Refusing to overwrite '{dest}': it is not a directory. "
                    "Pass a different `--output-dir`."
                )
            if not (dest / CONVERSION_MARKER).exists() and any(dest.iterdir()):
                raise ModelconverterException(
                    f"Refusing to overwrite '{dest}': it is not empty and "
                    "does not hold conversion results. Pass a different "
                    "`--output-dir`."
                )
            shutil.rmtree(dest)
        return dest
    return OUTPUTS_DIR / f"{name}_to_{target.name.lower()}_{date}"


def init_dirs() -> None:
    for p in [CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, CALIBRATION_DIR]:
        logger.debug(f"Creating {p}")
        p.mkdir(parents=True, exist_ok=True)


def get_configs(
    target: Target,
    path: str | None,
    opts: list[str] | dict[str, Any] | None = None,
) -> tuple[Config, NNArchiveConfig | None, str | None]:
    """Set up the configuration.

    Args:
        target: Conversion target the config is built for.
        path: Path to the configuration file or NN Archive.
        opts: Optional CLI overrides of the config file.

    Returns:
        Tuple of the parsed modelconverter `Config`, `NNArchiveConfig`
        and the main stage key.

    """
    opts = opts or []
    # `infer` parses a second config after `convert` has returned, in the same
    # process. Start from the default base so a directory left behind by the
    # previous config cannot resolve this one's relative paths.
    set_input_base(None)
    if isinstance(opts, list):
        if len(opts) % 2 != 0:
            raise ValueError(
                "Invalid number of overrides. See --help for more information."
            )
        overrides: Params = {
            opts[i]: opts[i + 1] for i in range(0, len(opts), 2)
        }
    else:
        overrides = opts
    if path is not None:
        path_ = resolve_path(path, MISC_DIR)
        if path_.is_dir() or is_nn_archive(path_):
            return process_nn_archive(target, path_, overrides)
        # Resolve files referenced *inside* the config (calibration data,
        # scripts, encodings, ...) relative to the config file's directory.
        set_input_base(path_.parent)
    cfg = Config.get_config(path, overrides)

    main_stage_key = None
    if len(cfg.stages) > 1:
        for key in cfg.stages:
            if "yolov8" in key and "seg" in key:
                logger.info(f"Detected main stage key: {key}")
                main_stage_key = key
                break
    else:
        main_stage_key = next(iter(cfg.stages.keys()))

    return cfg, None, main_stage_key


def extract_preprocessing(
    cfg: Config,
) -> tuple[Config, dict[str, PreprocessingBlock]]:
    if len(cfg.stages) > 1:
        raise ValueError(
            "Only single-stage models are supported with NN archive."
        )
    stage_cfg = next(iter(cfg.stages.values()))
    preprocessing = {}
    for inp in stage_cfg.inputs:
        mean = inp.mean_values
        scale = inp.scale_values
        encoding = inp.encoding
        layout = inp.layout

        if inp.is_raw_input:
            if mean is not None or scale is not None:
                preprocessing[inp.name] = PreprocessingBlock(
                    mean=mean,
                    scale=scale,
                    reverse_channels=None,
                    interleaved_to_planar=None,
                    dai_type=None,
                )
        else:
            dai_type = encoding.to.value
            if inp.data_type == DataType.FLOAT16:
                channel_type = "F16F16F16"
            else:
                channel_type = "888"
            dai_type += channel_type
            dai_type += "i" if layout == "NHWC" else "p"

            preprocessing[inp.name] = PreprocessingBlock(
                mean=mean or [0, 0, 0],
                scale=scale or [1, 1, 1],
                reverse_channels=encoding.to == Encoding.RGB,
                interleaved_to_planar=layout == "NHWC",
                dai_type=dai_type,
            )

        inp.mean_values = None
        inp.scale_values = None
        inp.encoding.from_ = Encoding.NONE
        inp.encoding.to = Encoding.NONE

    return cfg, preprocessing


def slug_to_id(
    slug: str, endpoint: Literal["models", "modelVersions", "modelInstances"]
) -> str:
    for is_public in [True, False]:
        with suppress(HTTPError):
            params = {
                "is_public": is_public,
                "slug": slug,
            }
            data = Request.get(f"{endpoint}/", params=params)
            if data:
                return data[0]["id"]
    raise ValueError(f"Model with slug '{slug}' not found.")
