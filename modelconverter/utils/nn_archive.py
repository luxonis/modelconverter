import json
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
from modelconverter.utils.metadata import get_metadata
from modelconverter.utils.shape import Shape
from modelconverter.utils.types import Target


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
        "calibration": "random",
    }

    for inp in archive_config.model.inputs:
        reverse = inp.preprocessing.reverse_channels
        if inp.input_type == InputType.IMAGE:
            if reverse:
                encoding = {"from": "RGB", "to": "BGR"}
            else:
                encoding = "BGR"
        else:
            encoding = "NONE"

        mean = inp.preprocessing.mean or [0, 0, 0]
        scale = inp.preprocessing.scale or [1, 1, 1]

        main_stage_config["inputs"].append(
            {
                "name": inp.name,
                "shape": inp.shape,
                "layout": inp.layout,
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

    main_stage_key = Path(archive_config.model.metadata.path).stem
    config = {
        "name": main_stage_key,
        "stages": {
            main_stage_key: main_stage_config,
        },
    }

    for head in archive_config.model.heads or []:
        postprocessor_path = getattr(head, "postprocessor_path", None)
        if postprocessor_path is not None:
            input_model_path = untar_path / postprocessor_path
            head_stage_config = {
                "input_model": str(input_model_path),
                "inputs": [],
                "outputs": [],
                "calibration": "random",
                "encoding": "NONE",
            }
            config["stages"][input_model_path.stem] = head_stage_config

    return Config.get_config(config, overrides), archive_config, main_stage_key


def modelconverter_config_to_nn(
    config: Config,
    target: Target,
    orig_nn: Optional[NNArchiveConfig],
    preprocessing: Dict[str, PreprocessingBlock],
    main_stage_key: str,
    model_path: Path,
) -> NNArchiveConfig:
    is_multistage = len(config.stages) > 1
    model_metadata = get_metadata(model_path)

    if main_stage_key is None:
        main_stage_key = next(iter(config.stages.keys()))

    cfg = config.stages[main_stage_key]
    archive_cfg = {
        "config_version": "1.0",
        "model": {
            "metadata": {
                "name": main_stage_key,
                "path": f"{main_stage_key}{target.suffix}",
            },
            "inputs": [],
            "outputs": [],
            "heads": orig_nn.model.heads if orig_nn else [],
        },
    }

    for inp in cfg.inputs:
        new_shape = model_metadata.input_shapes[inp.name]
        # new_dtype = model_metadata.input_dtypes[inp.name]
        if inp.shape is not None:
            layout = inp.shape.guess_new_layout(new_shape).layout
        else:
            layout = Shape._default_layout(new_shape)

        archive_cfg["model"]["inputs"].append(
            {
                "name": inp.name,
                "shape": new_shape,
                "layout": "".join(layout),
                # "dtype": new_dtype.value,
                "dtype": inp.data_type.value,
                # "dtype": "float32",
                "input_type": "image",
                "preprocessing": {
                    "mean": [0 for _ in inp.mean_values]
                    if inp.mean_values
                    else None,
                    "scale": [1 for _ in inp.scale_values]
                    if inp.scale_values
                    else None,
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                },
            }
        )
    for out in cfg.outputs:
        new_shape = model_metadata.output_shapes[out.name]
        new_dtype = model_metadata.output_dtypes[out.name]
        if out.shape is not None:
            layout = out.shape.guess_new_layout(new_shape).layout
        else:
            layout = Shape._default_layout(new_shape)

        archive_cfg["model"]["outputs"].append(
            {
                "name": out.name,
                "shape": new_shape,
                "layout": "".join(layout),
                # "dtype": new_dtype.value,
                "dtype": out.data_type.value,
                # "dtype": "float32",
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
        head.postprocessor_path = f"{post_stage_key}{target.suffix}"
    return archive
