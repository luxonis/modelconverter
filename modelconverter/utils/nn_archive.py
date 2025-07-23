import json
import tarfile
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from luxonis_ml.nn_archive import ArchiveGenerator
from luxonis_ml.nn_archive.config import CONFIG_VERSION
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import (
    Input as NNArchiveInput,
)
from luxonis_ml.nn_archive.config_building_blocks import (
    InputType,
    PreprocessingBlock,
)

from modelconverter.utils.config import BlobBaseConfig, Config, TargetConfig
from modelconverter.utils.constants import MISC_DIR
from modelconverter.utils.layout import guess_new_layout, make_default_layout
from modelconverter.utils.metadata import Metadata, get_metadata
from modelconverter.utils.types import DataType, Encoding, Target


def get_archive_input(cfg: NNArchiveConfig, name: str) -> NNArchiveInput:
    for inp in cfg.model.inputs:
        if inp.name == name:
            return inp
    raise ValueError(f"Input {name} not found in the archive config")


def process_nn_archive(
    path: Path, overrides: dict[str, Any] | None
) -> tuple[Config, NNArchiveConfig, str]:
    """Extracts the archive from tar and parses its config.

    @type path: Path
    @param path: Path to the archive.
    @type overrides: Optional[Dict[str, Any]]
    @param overrides: Config overrides.
    @rtype: Tuple[Config, NNArchiveConfig, str]
    @return: Tuple of the parsed config, NNArchiveConfig and the main
        stage key.
    """

    untar_path = MISC_DIR / path.stem
    if path.is_dir():
        untar_path = path
    elif tarfile.is_tarfile(path):
        if untar_path.suffix == ".tar":
            untar_path = MISC_DIR / untar_path.stem

        def safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
            """Filter members to prevent path traversal attacks."""
            safe_files = []
            for member in tar.getmembers():
                # Normalize path and ensure it's within the extraction folder
                if not member.name.startswith("/") and ".." not in member.name:
                    safe_files.append(member)
                else:
                    logger.warning(f"Skipping unsafe file: {member.name}")
            return safe_files

        with tarfile.open(path, mode="r") as tf:
            for member in safe_members(tf):
                tf.extract(member, path=untar_path)

    else:
        raise RuntimeError(f"Unknown NN Archive path: `{path}`")

    if not (untar_path / "config.json").exists():
        raise RuntimeError(f"NN Archive config not found in `{untar_path}`")

    with open(untar_path / "config.json") as f:
        archive_config = NNArchiveConfig(**json.load(f))

    main_stage_config = {
        "name": archive_config.model.metadata.name,
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
                "encoding": encoding
                if isinstance(encoding, dict)
                else {"from": encoding, "to": encoding},
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

    stages = {}

    for head in archive_config.model.heads or []:
        postprocessor_path = getattr(head.metadata, "postprocessor_path", None)
        if postprocessor_path is not None:
            input_model_path = untar_path / postprocessor_path
            head_stage_config = {
                "input_model": str(input_model_path),
                "inputs": [],
                "outputs": [],
                "encoding": {"from": "NONE", "to": "NONE"},
            }
            stages[input_model_path.stem] = head_stage_config

    if stages:
        main_stage_key = main_stage_config.pop("name")
        config = {
            "name": main_stage_key,
            "stages": {
                main_stage_key: main_stage_config,
                **stages,
            },
        }
    else:
        config = main_stage_config
        main_stage_key = config["name"]

    return Config.get_config(config, overrides), archive_config, main_stage_key


def modelconverter_config_to_nn(
    config: Config,
    model_name: Path,
    orig_nn: NNArchiveConfig | None,
    preprocessing: dict[str, PreprocessingBlock],
    main_stage_key: str,
    model_path: Path,
    target: Target,
) -> NNArchiveConfig:
    is_multistage = len(config.stages) > 1
    model_metadata = get_metadata(model_path)

    cfg = config.stages[main_stage_key]
    target_cfg = cfg.get_target_config(target)

    # TODO: This might be more complicated for Hailo

    compress_to_fp16 = getattr(target_cfg, "compress_to_fp16", False)
    disable_calibration = target_cfg.disable_calibration

    match target, compress_to_fp16, disable_calibration:
        case Target.RVC2, True, _:
            precision = DataType.FLOAT16

        case Target.RVC2, False, _:
            precision = DataType.FLOAT32

        case Target.RVC3 | Target.RVC4, True, True:
            precision = DataType.FLOAT16

        case Target.RVC3 | Target.RVC4, False, True:
            precision = DataType.FLOAT32

        case Target.RVC3 | Target.RVC4, _, False:
            precision = DataType.INT8

        case Target.HAILO, _, _:
            precision = DataType.INT8

    archive_cfg = {
        "config_version": CONFIG_VERSION,
        "model": {
            "metadata": {
                "name": model_name.stem,
                "path": str(model_name),
                "precision": precision.value,
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

        dtype = _get_io_dtype(
            target,
            inp.name,
            model_metadata,
            target_cfg,
            mode="input",
        )

        archive_cfg["model"]["inputs"].append(
            {
                "name": inp.name,
                "shape": new_shape,
                "layout": layout,
                "dtype": dtype,
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
            try:
                layout = guess_new_layout(out.layout, out.shape, new_shape)
            except ValueError as e:
                layout = make_default_layout(new_shape)
                logger.warning(
                    f"Unable to infer layout for layer '{out.name}': {e}. "
                    f"The original shape was `{out.shape}`, which is incompatible "
                    f"with the shape of the converted model: `{new_shape}`. "
                    f"Changing the layout of the converted model to `{layout}`. "
                )
        else:
            layout = make_default_layout(new_shape)

        dtype = _get_io_dtype(
            target,
            out.name,
            model_metadata,
            target_cfg,
            mode="output",
        )
        archive_cfg["model"]["outputs"].append(
            {
                "name": out.name,
                "shape": new_shape,
                "layout": layout,
                "dtype": dtype,
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
        post_stage_key = next(
            key for key in config.stages if key != main_stage_key
        )
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


def generate_archive(
    target: Target,
    cfg: Config,
    main_stage: str,
    out_models: list[Path],
    output_path: Path,
    archive_cfg: NNArchiveConfig | None,
    preprocessing: dict[str, PreprocessingBlock],
    inference_model_path: Path,
    archive_name: str | None,
) -> Path:
    logger.info("Converting to NN archive")
    if len(out_models) > 1:
        model_name = f"{main_stage}{out_models[0].suffix}"
    else:
        model_name = out_models[0].name
    nn_archive = modelconverter_config_to_nn(
        cfg,
        Path(model_name),
        archive_cfg,
        preprocessing,
        main_stage,
        inference_model_path,
        target,
    )
    generator = ArchiveGenerator(
        archive_name=f"{archive_name or cfg.name}.{target.value.lower()}",
        save_path=str(output_path),
        cfg_dict=nn_archive.model_dump(),
        executables_paths=[
            *out_models,
            output_path / "buildinfo.json",
        ],
    )
    archive = generator.make_archive()
    logger.info(f"Model exported to {archive}")
    return archive


def _get_io_dtype(
    target: Target,
    name: str,
    metadata: Metadata,
    cfg: TargetConfig,
    *,
    mode: Literal["input", "output"],
) -> str:
    if mode == "input":
        dtypes = metadata.input_dtypes
    else:
        dtypes = metadata.output_dtypes
    if target in {Target.RVC2, Target.RVC3}:
        compile_tool_args: list[str] = getattr(cfg, "compile_tool_args", [])
        assert isinstance(cfg, BlobBaseConfig)
        # -iop is in a form of '-iop "<name1>:<dtype>,<name2>:<dtype2>"'
        if "-iop" in compile_tool_args:
            idx = compile_tool_args.index("-iop")
            for value in compile_tool_args[idx + 1].split(","):
                value = value.strip()
                n, d = value.split(":")
                if n == name:
                    blob_dtype = d.upper()
                    return DataType.from_ir_ie_dtype(
                        blob_dtype
                    ).as_nn_archive_dtype()

        elif mode == "input" and "-ip" in compile_tool_args:
            idx = compile_tool_args.index("-ip")
        elif mode == "output" and "-op" in compile_tool_args:
            idx = compile_tool_args.index("-op")
        else:
            return dtypes[name].as_nn_archive_dtype()

        blob_dtype = compile_tool_args[idx + 1].upper()
        return DataType.from_ir_ie_dtype(blob_dtype).as_nn_archive_dtype()

    return dtypes[name].as_nn_archive_dtype()
