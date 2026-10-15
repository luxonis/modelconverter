"""Helpers shared by the ``modelconverter`` CLI commands.

Turns what the user typed on the command line into what the platform
packages expect: the directory a run writes its results to, the parsed
configuration (from a config file or an NN Archive), and the
preprocessing that is handed to the NN Archive instead of being baked
into the model.
"""

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from luxonis_ml.nn_archive import is_nn_archive
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import PreprocessingBlock
from luxonis_ml.typing import Params

from modelconverter.utils import (
    ModelconverterException,
    make_dai_type,
    process_nn_archive,
    resolve_path,
    sanitize_net_name,
)
from modelconverter.utils.config import (
    Config,
    InputConfig,
    broadcast_preprocessing_values,
)
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    CONFIGS_DIR,
    CONVERSION_MARKER,
    HOST_OUTPUT_DIR_ENV_VAR,
    MISC_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    in_docker,
)
from modelconverter.utils.filesystem_utils import set_input_base
from modelconverter.utils.layout import is_interleaved_image_layout
from modelconverter.utils.preprocessing import CalibrationPreprocessing
from modelconverter.utils.types import Encoding, Platform


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


def display_output_path(path: Path) -> str:
    """Return the host-visible representation of an output path when available.

    Args:
        path: Artifact path produced by the current conversion environment.

    Returns:
        Host-visible path for a Docker output when mapping context is
        available, otherwise the original path representation.

    """
    host_output_dir = os.environ.get(HOST_OUTPUT_DIR_ENV_VAR)
    if not in_docker() or not host_output_dir:
        return str(path)

    try:
        relative = path.relative_to(OUTPUTS_DIR)
    except ValueError:
        return str(path)

    if ".." in relative.parts:
        return str(path)

    return _join_display_path(host_output_dir, relative)


def get_output_dir_name(
    platform: Platform, name: str, output_dir: str | None
) -> Path:
    """Determine the directory the conversion writes its results to.

    An existing destination is cleared first, but only when it is a
    directory that a previous conversion produced -- one holding a
    `CONVERSION_MARKER` file -- or an empty one.

    Args:
        platform: Platform the model is converted for, used in the
            generated name.
        name: Name of the model, sanitized before use.
        output_dir: Directory named by the user, resolved under
            `OUTPUTS_DIR`. If ``None``, a directory named
            ``<name>_to_<platform>_<timestamp>`` is used instead.

    Returns:
        Path to the output directory. It is not created here.

    Raises:
        ModelconverterException: If, in a containerized run,
            ``output_dir`` is empty, absolute or contains ``..``; or if
            the destination exists but is not a directory, or is a
            non-empty directory that does not hold the results of a
            previous conversion.

    """
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
    return OUTPUTS_DIR / f"{name}_to_{platform.name.lower()}_{date}"


def init_dirs() -> None:
    """Create the directories a conversion reads from and writes to."""
    for p in [CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, CALIBRATION_DIR]:
        logger.debug(f"Creating {p}")
        p.mkdir(parents=True, exist_ok=True)


def get_configs(
    platform: Platform,
    path: str | None,
    opts: list[str] | Params | None = None,
) -> tuple[Config, NNArchiveConfig | None, str | None]:
    """Set up the configuration.

    Args:
        platform: Platform the config is built for.
        path: Path to the configuration file or NN Archive. If
            ``None``, the config is built from the overrides alone.
        opts: Optional CLI overrides of the config file. Either a
            mapping, or a flat list alternating keys and values.

    Returns:
        Tuple of the parsed modelconverter `Config`, the
        ``NNArchiveConfig`` if the input was an NN Archive and ``None``
        otherwise, and the key of the main stage -- ``None`` for a
        multi-stage config in which no main stage was recognized.

    Raises:
        ValueError: If ``opts`` is a list of odd length.

    """
    # `infer` parses a second config after `convert` has returned, in the same
    # process. Start from the default base so a directory left behind by the
    # previous config cannot resolve this one's relative paths.
    set_input_base(None)
    overrides = _parse_overrides(opts)
    if path is not None:
        path_ = resolve_path(path, MISC_DIR)
        if path_.is_dir() or is_nn_archive(path_):
            return process_nn_archive(platform, path_, overrides)
        # Resolve files referenced *inside* the config (calibration data,
        # scripts, encodings, ...) relative to the config file's directory.
        set_input_base(path_.parent)
    cfg = Config.get_config(path, overrides)
    return cfg, None, _find_main_stage(cfg)


def _parse_overrides(opts: list[str] | Params | None) -> Params:
    """Turn the CLI overrides into a mapping of keys to values."""
    if not opts:
        return {}
    if not isinstance(opts, list):
        return opts
    if len(opts) % 2 != 0:
        raise ValueError(
            "Invalid number of overrides. See --help for more information."
        )
    return dict(zip(opts[::2], opts[1::2], strict=True))


def _find_main_stage(cfg: Config) -> str | None:
    """Return the only stage, or the YOLOv8 segmentation stage."""
    if len(cfg.stages) == 1:
        return next(iter(cfg.stages))
    for key in cfg.stages:
        if "yolov8" in key and "seg" in key:
            logger.info(f"Detected main stage key: {key}")
            return key
    return None


def extract_preprocessing(
    cfg: Config,
) -> tuple[Config, dict[str, PreprocessingBlock]]:
    """Move the preprocessing out of the config into archive blocks.

    Mean values, scale values and the color encoding are cleared on
    every input of the config -- so they are not baked into the
    converted model -- and returned as ``PreprocessingBlock`` objects to
    be stored in the NN Archive instead. A raw input only gets a block
    when it has mean or scale values.

    Args:
        cfg: Single-stage config to take the preprocessing from. Its
            inputs are modified in place.

    Returns:
        Tuple of the modified config and the preprocessing blocks keyed
        by input name.

    Raises:
        ValueError: If the config has more than one stage.

    """
    if len(cfg.stages) > 1:
        raise ValueError(
            "Only single-stage models are supported with NN archive."
        )
    stage_cfg = next(iter(cfg.stages.values()))
    preprocessing = {}
    for inp in stage_cfg.inputs:
        inp.validate_input_contract()
        mean = _broadcast(inp.mean_values, inp.channel_count)
        scale = _broadcast(inp.scale_values, inp.channel_count)

        # Remember the preprocessing for calibration before it is cleared.
        inp._calibration_preprocessing = CalibrationPreprocessing(
            encoding_from=inp.encoding.from_,
            encoding_to=inp.encoding.to,
            mean_values=None if mean is None else tuple(mean),
            scale_values=None if scale is None else tuple(scale),
            data_type=inp.data_type,
            is_image=not inp.is_raw_input,
        )

        block = _archive_preprocessing(inp, mean, scale)
        if block is not None:
            preprocessing[inp.name] = block

        inp.mean_values = None
        inp.scale_values = None
        inp.encoding.from_ = Encoding.NONE
        inp.encoding.to = Encoding.NONE

    return cfg, preprocessing


def _broadcast(
    values: list[float] | None, channels: int | None
) -> list[float] | None:
    """Broadcast optional preprocessing values to the channel count."""
    if values is None:
        return None
    return broadcast_preprocessing_values(values, channels)


def _archive_preprocessing(
    inp: InputConfig, mean: list[float] | None, scale: list[float] | None
) -> PreprocessingBlock | None:
    """Build the archive preprocessing block of one input.

    A raw input without mean and scale values needs no block.
    """
    if inp.is_raw_input:
        if mean is None and scale is None:
            return None
        return PreprocessingBlock.model_validate(
            {
                "mean": mean,
                "scale": scale,
                "reverse_channels": None,
                "interleaved_to_planar": None,
                "dai_type": None,
            }
        )

    # Once preprocessing is externalized, the converted model is fed
    # directly in the format expected by the source graph.
    encoding = inp.encoding.from_
    identity_value_count = 1 if encoding == Encoding.GRAY else 3
    return PreprocessingBlock.model_validate(
        {
            "mean": [0.0] * identity_value_count if mean is None else mean,
            "scale": [1.0] * identity_value_count if scale is None else scale,
            "reverse_channels": encoding == Encoding.RGB,
            "interleaved_to_planar": is_interleaved_image_layout(inp.layout),
            "dai_type": make_dai_type(encoding, inp.data_type, inp.layout),
        }
    )


def _join_display_path(root: str, relative: Path) -> str:
    relative_display = relative.as_posix()
    if relative_display == ".":
        return root

    use_backslash = "\\" in root and "/" not in root
    separator = "\\" if use_backslash else "/"
    if use_backslash:
        relative_display = relative_display.replace("/", separator)

    clean_root = root.rstrip("/\\")
    if not clean_root:
        return f"{separator}{relative_display}"
    return f"{clean_root}{separator}{relative_display}"
