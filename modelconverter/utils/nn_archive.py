import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from luxonis_ml.nn_archive.config import CONFIG_VERSION
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import (
    Input as NNArchiveInput,
)
from luxonis_ml.nn_archive.config_building_blocks import (
    InputType,
    PreprocessingBlock,
)

from modelconverter.utils.config import Config
from modelconverter.utils.constants import MISC_DIR
from modelconverter.utils.layout import guess_new_layout, make_default_layout
from modelconverter.utils.metadata import get_metadata
from modelconverter.utils.types import DataType, Encoding

logger = logging.getLogger(__name__)


def get_archive_input(cfg: NNArchiveConfig, name: str) -> NNArchiveInput:
    for inp in cfg.model.inputs:
        if inp.name == name:
            return inp
    raise ValueError(f"Input {name} not found in the archive config")


def process_nn_archive(
    path: Path, overrides: Optional[Dict[str, Any]]
) -> Tuple[Config, NNArchiveConfig, str]:
    """Extracts the archive from tar and parses its config.

    @type path: Path
    @param path: Path to the archive.
    @type overrides: Optional[Dict[str, Any]]
    @param overrides: Config overrides.
    @rtype: Tuple[Config, NNArchiveConfig, str]
    @return: Tuple of the parsed config, NNArchiveConfig and the main stage key.
    """

    untar_path = MISC_DIR / path.stem
    if path.is_dir():
        untar_path = path
    elif tarfile.is_tarfile(path):
        if untar_path.suffix == ".tar":
            untar_path = MISC_DIR / untar_path.stem
        with tarfile.open(path) as tar:
            tar.extractall(untar_path)
    else:
        raise RuntimeError(f"Unknown NN Archive path: `{path}`")

    if not (untar_path / "config.json").exists():
        raise RuntimeError(f"NN Archive config not found in `{untar_path}`")

    with open(untar_path / "config.json") as f:
        archive_config = NNArchiveConfig(**json.load(f))

    main_stage_config = {
        "input_model": str(untar_path / archive_config.model.metadata.path),
        "inputs": [],
        "outputs": [],
    }

    for inp in archive_config.model.inputs:
        reverse = inp.preprocessing.reverse_channels
        interleaved_to_planar = inp.preprocessing.interleaved_to_planar
        dai_type = inp.preprocessing.dai_type

        layout = inp.layout
        encoding = "NONE"
        if inp.input_type == InputType.IMAGE:
            if dai_type is not None:
                if (reverse and dai_type.startswith("BGR")) or (
                    reverse is False and dai_type.startswith("RGB")
                ):
                    logger.warning(
                        "'reverse_channels' and 'dai_type' are conflicting, using dai_type"
                    )

                if dai_type.startswith("RGB"):
                    encoding = {"from": "RGB", "to": "BGR"}
                elif dai_type.startswith("BGR"):
                    encoding = "BGR"
                elif dai_type.startswith("GRAY"):
                    encoding = "GRAY"
                else:
                    logger.warning("unknown dai_type, using RGB888p")
                    encoding = {"from": "RGB", "to": "BGR"}

                if (interleaved_to_planar and dai_type.endswith("p")) or (
                    interleaved_to_planar is False and dai_type.endswith("i")
                ):
                    logger.warning(
                        "'interleaved_to_planar' and 'dai_type' are conflicting, using dai_type"
                    )
                if dai_type.endswith("i"):
                    layout = "NHWC"
                elif dai_type.endswith("p"):
                    layout = "NCHW"
            else:
                if reverse is not None:
                    logger.warning(
                        "'reverse_channels' flag is deprecated and will be removed in the future, use 'dai_type' instead"
                    )
                    if reverse:
                        encoding = {"from": "RGB", "to": "BGR"}
                    else:
                        encoding = "BGR"
                else:
                    encoding = {"from": "RGB", "to": "BGR"}

                if interleaved_to_planar is not None:
                    logger.warning(
                        "'interleaved_to_planar' flag is deprecated and will be removed in the future, use 'dai_type' instead"
                    )
                    if interleaved_to_planar:
                        layout = "NHWC"
                    else:
                        layout = "NCHW"
            channels = (
                inp.shape[layout.index("C")]
                if layout and "C" in layout
                else None
            )
            if channels and channels == 1:
                encoding = "GRAY"

        mean = inp.preprocessing.mean or [0, 0, 0]
        scale = inp.preprocessing.scale or [1, 1, 1]

        main_stage_config["inputs"].append(
            {
                "name": inp.name,
                "shape": inp.shape,
                "layout": layout,
                "data_type": inp.dtype.value,
                "mean_values": mean,
                "scale_values": scale,
                "encoding": encoding,
            }
        )

    for out in archive_config.model.outputs:
        main_stage_config["outputs"].append(
            {
                "name": out.name,
                "shape": out.shape,
                "layout": out.layout,
                "data_type": out.dtype.value,
            }
        )

    main_stage_key = archive_config.model.metadata.name
    config = {
        "name": main_stage_key,
        "stages": {
            main_stage_key: main_stage_config,
        },
    }

    for head in archive_config.model.heads or []:
        postprocessor_path = getattr(head.metadata, "postprocessor_path", None)
        if postprocessor_path is not None:
            input_model_path = untar_path / postprocessor_path
            head_stage_config = {
                "input_model": str(input_model_path),
                "inputs": [],
                "outputs": [],
                "encoding": "NONE",
            }
            config["stages"][input_model_path.stem] = head_stage_config

    return Config.get_config(config, overrides), archive_config, main_stage_key


def modelconverter_config_to_nn(
    config: Config,
    model_name: Path,
    orig_nn: Optional[NNArchiveConfig],
    preprocessing: Dict[str, PreprocessingBlock],
    main_stage_key: str,
    model_path: Path,
) -> NNArchiveConfig:
    is_multistage = len(config.stages) > 1
    model_metadata = get_metadata(model_path)

    cfg = config.stages[main_stage_key]

    archive_cfg = {
        "config_version": CONFIG_VERSION,
        "model": {
            "metadata": {
                "name": model_name.stem,
                "path": str(model_name),
            },
            "inputs": [],
            "outputs": [],
            "heads": orig_nn.model.heads if orig_nn else [],
        },
    }

    for inp in cfg.inputs:
        new_shape = model_metadata.input_shapes[inp.name]
        if inp.shape is not None and not any(s == 0 for s in inp.shape):
            assert inp.layout is not None
            layout = guess_new_layout(inp.layout, inp.shape, new_shape)
        else:
            layout = make_default_layout(new_shape)
        dai_type = inp.encoding.to.value
        if inp.data_type == DataType.FLOAT16:
            type = "F16F16F16"
        else:
            type = "888"
        dai_type += type
        dai_type += "i" if layout == "NHWC" else "p"

        archive_cfg["model"]["inputs"].append(
            {
                "name": inp.name,
                "shape": new_shape,
                "layout": layout,
                "dtype": inp.data_type.value,
                "input_type": "image",
                "preprocessing": {
                    "mean": [0 for _ in inp.mean_values]
                    if inp.mean_values
                    else None,
                    "scale": (
                        [1 for _ in inp.scale_values]
                        if inp.scale_values
                        else None
                    ),
                    "reverse_channels": inp.encoding.to == Encoding.RGB,
                    "interleaved_to_planar": layout == "NHWC",
                    "dai_type": dai_type,
                },
            }
        )
    for out in cfg.outputs:
        new_shape = model_metadata.output_shapes[out.name]
        if out.shape is not None and not any(s == 0 for s in out.shape):
            assert out.layout is not None
            layout = guess_new_layout(out.layout, out.shape, new_shape)
        else:
            layout = make_default_layout(new_shape)

        archive_cfg["model"]["outputs"].append(
            {
                "name": out.name,
                "shape": new_shape,
                "layout": layout,
                "dtype": out.data_type.value,
            }
        )

    archive = NNArchiveConfig(**archive_cfg)

    for name, block in preprocessing.items():
        nn_inp = get_archive_input(archive, name)
        nn_inp.preprocessing = block

    if is_multistage:
        if len(config.stages) > 2:
            raise NotImplementedError(
                "Only 2-stage models are supported with NN Archive for now."
            )
        post_stage_key = [
            key for key in config.stages if key != main_stage_key
        ][0]
        if not archive.model.heads:
            raise ValueError(
                "Multistage NN Archives must sxpecify 1 head in the archive config"
            )
        head = archive.model.heads[0]
        head.metadata.postprocessor_path = (
            f"{post_stage_key}{model_name.suffix}"
        )
    return archive


def archive_from_model(model_path: Path) -> NNArchiveConfig:
    metadata = get_metadata(model_path)

    archive_cfg = {
        "config_version": CONFIG_VERSION,
        "model": {
            "metadata": {
                "name": model_path.stem,
                "path": model_path.name,
            },
            "inputs": [],
            "outputs": [],
            "heads": [],
        },
    }

    for name, shape in metadata.input_shapes.items():
        archive_cfg["model"]["inputs"].append(
            {
                "name": name,
                "shape": shape,
                "layout": make_default_layout(shape),
                "dtype": metadata.input_dtypes[name].value,
                "input_type": "image",
                "preprocessing": {
                    "mean": None,
                    "scale": None,
                    "reverse_channels": None,
                    "interleaved_to_planar": None,
                    "dai_type": None,
                },
            }
        )

    for name, shape in metadata.output_shapes.items():
        archive_cfg["model"]["outputs"].append(
            {
                "name": name,
                "shape": shape,
                "layout": make_default_layout(shape),
                "dtype": metadata.output_dtypes[name].value,
            }
        )

    return NNArchiveConfig(**archive_cfg)
