"""Translation between modelconverter configs and NN Archives.

An NN Archive is the Luxonis packaging format: a tar holding one or
more model files next to a ``config.json`` that describes their inputs,
outputs, preprocessing and heads. It sits on both ends of a conversion
-- an archive can be handed to the converter, in which case it is
unpacked and turned into a `Config` here, and the converted models are
packed back into a new archive whose config this module builds.
"""

import json
import os
import tarfile
from itertools import pairwise
from pathlib import Path
from typing import Literal

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
from luxonis_ml.typing import Params
from pydantic import BaseModel

from modelconverter.utils.config import (
    BlobBaseConfig,
    Config,
    InputConfig,
    OutputConfig,
    PlatformConfig,
    broadcast_preprocessing_values,
)
from modelconverter.utils.constants import MISC_DIR
from modelconverter.utils.layout import (
    guess_new_layout,
    is_image_input_shape,
    is_interleaved_image_layout,
    make_default_layout,
)
from modelconverter.utils.metadata import Metadata, get_metadata
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    Platform,
    QuantizationMode,
    ResizeMethod,
)


def get_archive_input(cfg: NNArchiveConfig, name: str) -> NNArchiveInput:
    """Look up an input of an archive config by name.

    Args:
        cfg: Archive config to search.
        name: Name of the input to look for.

    Returns:
        The matching archive input.

    Raises:
        ValueError: If the config declares no input of that name.

    """
    for inp in cfg.model.inputs:
        if inp.name == name:
            return inp
    raise ValueError(f"Input {name} not found in the archive config")


def process_nn_archive(
    platform: Platform, path: Path, overrides: Params | None
) -> tuple[Config, NNArchiveConfig, str]:
    """Read an NN Archive and parse its config.

    Args:
        platform: Platform the config is built for.
        path: Path to the archive. It is either a tar file to unpack, or
            a directory that already holds an unpacked archive.
        overrides: CLI overrides applied on top of the archive's own
            configuration.

    Returns:
        Tuple of the parsed `Config`, ``NNArchiveConfig`` and the main
        stage key.

    Raises:
        RuntimeError: If the path is neither a directory nor a tar file,
            or if the archive holds no ``config.json``.

    """
    untar_path = _unpack_archive(path)
    if not (untar_path / "config.json").exists():
        raise RuntimeError(f"NN Archive config not found in `{untar_path}`")

    with open(untar_path / "config.json") as f:
        archive_config = NNArchiveConfig(**json.load(f))
    package_name = _package_name(path)

    main_stage_key = archive_config.model.metadata.name
    main_stage_config: Params = {
        "input_model": str(untar_path / archive_config.model.metadata.path),
    }

    if (
        platform is Platform.RVC4
        and (p := untar_path / "encodings.json").exists()
    ):
        logger.info("Using custom `encodings.json` from the NN Archive.")
        main_stage_config["rvc4"] = {
            "encodings": json.loads(p.read_text()),
        }

    main_stage_config["inputs"] = [
        _archive_input_config(inp) for inp in archive_config.model.inputs
    ]
    main_stage_config["outputs"] = [
        {
            "name": out.name,
            "shape": out.shape,
            "layout": out.layout,
            "data_type": out.dtype.value,
        }
        for out in archive_config.model.outputs
    ]
    stages = _postprocessor_stages(archive_config, untar_path)

    # Archive stages already have identities independent of the package name.
    config: Params = {
        "stages": {
            main_stage_key: main_stage_config,
            **stages,
        },
    }

    root_overrides, stage_overrides = _split_overrides(
        overrides, single_stage=not stages
    )
    # Single-stage shorthand must update the imported inputs/outputs, not
    # become root defaults that are ignored for fields already in the stage.
    Config._merge_overrides(main_stage_config, stage_overrides)
    Config._merge_overrides(config, root_overrides)
    # Resolve the archive default after overrides have been parsed, so
    # omitted names and null values follow the same fallback.
    if config.get("name") is None:
        config["name"] = package_name
    cfg = Config.model_validate(config)
    # Apply archive defaults after all root/stage/input overrides have been resolved.
    _apply_archive_resize_modes(
        cfg.stages[main_stage_key].inputs, archive_config
    )
    return cfg, archive_config, main_stage_key


_ARCHIVE_SUFFIXES = (
    ".tar.xz",
    ".tar.gz",
    ".tar.bz2",
    ".tar",
    ".tgz",
    ".txz",
    ".tbz2",
)


def _unpack_archive(path: Path) -> Path:
    """Return the directory of an archive.

    A tar file is unpacked into `MISC_DIR` first. A directory is taken
    as an unpacked archive.
    """
    if path.is_dir():
        return path
    if not tarfile.is_tarfile(path):
        raise RuntimeError(f"Unknown NN Archive path: `{path}`")

    untar_path = MISC_DIR / path.stem
    if untar_path.suffix == ".tar":
        untar_path = MISC_DIR / untar_path.stem
    with tarfile.open(path, mode="r") as tf:
        for member in _safe_members(tf):
            tf.extract(member, path=untar_path)
    return untar_path


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Filter members to prevent path traversal attacks."""
    safe_files = []
    for member in tar.getmembers():
        # Normalize path and ensure it's within the extraction folder
        if not member.name.startswith("/") and ".." not in member.name:
            safe_files.append(member)
        else:
            logger.warning(f"Skipping unsafe file: {member.name}")
    return safe_files


def _package_name(path: Path) -> str:
    """Name the package after the archive.

    Strip recognized archive/model suffixes, preserving other package
    dots.
    """
    package_name = Path(os.path.abspath(path)).name  # noqa: PTH100
    if path.is_dir():
        return package_name
    suffix = next(
        (s for s in _ARCHIVE_SUFFIXES if package_name.lower().endswith(s)),
        None,
    )
    if suffix is None:
        return package_name
    package_name = package_name[: -len(suffix)]
    try:
        InputFileType.from_path(package_name.lower())
    except ValueError:
        return package_name
    return Path(package_name).stem


def _archive_input_config(inp: NNArchiveInput) -> Params:
    """Turn an input of an archive into a modelconverter input config.

    An image input without mean or scale values gets identity values.
    """
    layout = inp.layout
    encoding: str | dict[str, str] = "NONE"
    if inp.input_type == InputType.IMAGE and is_image_input_shape(
        inp.shape, layout, allow_batchless=True
    ):
        encoding, layout = _archive_image_encoding(inp)

    color = encoding if isinstance(encoding, str) else encoding["from"]
    mean = inp.preprocessing.mean
    if mean is None:
        mean = _identity_values(color, 0)
    scale = inp.preprocessing.scale
    if scale is None:
        scale = _identity_values(color, 1)

    return {
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


def _identity_values(encoding: str, value: int) -> list[int] | None:
    """Repeat ``value`` once per channel of a color or a gray encoding."""
    if encoding in {"RGB", "BGR"}:
        return [value] * 3
    if encoding == "GRAY":
        return [value]
    return None


def _archive_image_encoding(
    inp: NNArchiveInput,
) -> tuple[str | dict[str, str], str | None]:
    """Derive the encoding and the layout of an image input of an archive.

    ``dai_type`` decides both. Without it, the deprecated
    ``reverse_channels`` flag decides the encoding. An input with one
    channel is gray in either case.
    """
    dai_type = inp.preprocessing.dai_type
    if dai_type is not None:
        encoding, layout = _dai_type_encoding(inp, dai_type)
    else:
        encoding, layout = _legacy_encoding(inp.preprocessing), inp.layout
    channels = (
        inp.shape[layout.index("C")] if layout and "C" in layout else None
    )
    if channels == 1:
        encoding = "GRAY"
    return encoding, layout


def _dai_type_encoding(
    inp: NNArchiveInput, dai_type: str
) -> tuple[str | dict[str, str], str | None]:
    """Derive the encoding and the layout of an input from its ``dai_type``.

    The layout follows the ``i`` (interleaved) or ``p`` (planar) suffix
    when the shape of the input fits it. The deprecated flags lose
    against ``dai_type``, with a warning.
    """
    preprocessing = inp.preprocessing
    reverse = preprocessing.reverse_channels
    if (reverse and dai_type.startswith("BGR")) or (
        reverse is False and dai_type.startswith("RGB")
    ):
        logger.warning(
            "'reverse_channels' and 'dai_type' are conflicting, using dai_type"
        )
    encoding = _dai_type_color(dai_type)

    interleaved_to_planar = preprocessing.interleaved_to_planar
    if (interleaved_to_planar and dai_type.endswith("p")) or (
        interleaved_to_planar is False and dai_type.endswith("i")
    ):
        logger.warning(
            "'interleaved_to_planar' and 'dai_type' are conflicting, using dai_type"
        )
    dai_layout = {"i": "NHWC", "p": "NCHW"}.get(dai_type[-1:])
    if dai_layout is not None and is_image_input_shape(inp.shape, dai_layout):
        return encoding, dai_layout
    return encoding, inp.layout


def _dai_type_color(dai_type: str) -> str | dict[str, str]:
    """Map the color order of a ``dai_type`` to an input encoding."""
    if dai_type.startswith("RGB"):
        return {"from": "RGB", "to": "BGR"}
    if dai_type.startswith("BGR"):
        return "BGR"
    if dai_type.startswith("GRAY"):
        return "GRAY"
    logger.warning("unknown dai_type, using RGB888p")
    return {"from": "RGB", "to": "BGR"}


def _legacy_encoding(
    preprocessing: PreprocessingBlock,
) -> str | dict[str, str]:
    """Derive the encoding of an input from the deprecated flags."""
    reverse = preprocessing.reverse_channels
    if reverse is None:
        encoding: str | dict[str, str] = {"from": "RGB", "to": "BGR"}
    else:
        logger.warning(
            "'reverse_channels' flag is deprecated and will be removed in the future, use 'dai_type' instead"
        )
        encoding = {"from": "RGB", "to": "BGR"} if reverse else "BGR"
    if preprocessing.interleaved_to_planar is not None:
        logger.warning(
            "'interleaved_to_planar' flag is deprecated and will be removed in the future, use 'dai_type' instead"
        )
    return encoding


def _postprocessor_stages(
    archive_config: NNArchiveConfig, untar_path: Path
) -> Params:
    """Make a stage for the postprocessor model of each head."""
    stages: Params = {}
    for head in archive_config.model.heads or []:
        postprocessor_path = getattr(head.metadata, "postprocessor_path", None)
        if postprocessor_path is None:
            continue
        input_model_path = untar_path / postprocessor_path
        stages[input_model_path.stem] = {
            "input_model": str(input_model_path),
            "inputs": [],
            "outputs": [],
            "encoding": {"from": "NONE", "to": "NONE"},
        }
    return stages


def _split_overrides(
    overrides: Params | None, *, single_stage: bool
) -> tuple[Params, Params]:
    """Split the overrides between the root config and the main stage.

    In a single-stage config, a key that is not a field of `Config`
    belongs to the stage.

    Returns:
        The root overrides and the stage overrides.
    """
    root_overrides: Params = {}
    stage_overrides: Params = {}
    for key, value in (overrides or {}).items():
        if single_stage and key.split(".", 1)[0] not in Config.model_fields:
            stage_overrides[key] = value
        else:
            root_overrides[key] = value
    return root_overrides, stage_overrides


def _apply_archive_resize_modes(
    inputs: list[InputConfig], archive_config: NNArchiveConfig
) -> None:
    """Take the archive resize mode for image inputs that set none."""
    original_inputs = {inp.name: inp for inp in archive_config.model.inputs}
    for inp in inputs:
        original = original_inputs.get(inp.name)
        if (
            original is None
            or inp.is_raw_input
            or inp.calibration.has_resize_method
        ):
            continue
        resize_mode = getattr(original.preprocessing, "resize_mode", None)
        if resize_mode is not None:
            inp.calibration.resize_method = ResizeMethod.from_nn_archive(
                resize_mode
            )


def _archive_precision(
    platform: Platform, platform_cfg: PlatformConfig
) -> DataType:
    """Derive the precision of a converted model from its platform config."""
    # TODO: This might be more complicated for Hailo
    if platform is Platform.HAILO:
        return DataType.INT8
    quantization_mode = getattr(platform_cfg, "quantization_mode", None)
    # RVC2 does not quantize, and RVC3 and RVC4 keep floats when calibration
    # is off.
    if platform is Platform.RVC2 or platform_cfg.disable_calibration:
        if quantization_mode in {None, QuantizationMode.CUSTOM}:
            fp16 = _compress_to_fp16(platform_cfg)
        else:
            fp16 = quantization_mode == QuantizationMode.FP16_STD
        return DataType.FLOAT16 if fp16 else DataType.FLOAT32
    if quantization_mode == QuantizationMode.INT16_STD:
        return DataType.INT16
    return DataType.INT8


def _compress_to_fp16(platform_cfg: PlatformConfig) -> bool:
    """Tell whether a config without a preset quantization mode keeps
    FP16.

    The ``compress_to_fp16`` option decides it. SNPE also runs in FP16
    with ``--float_bitwidth 16`` for the conversion and
    ``--use_float_io`` for the graph preparation.
    """
    onnx_args = getattr(platform_cfg, "snpe_onnx_to_dlc_args", [])
    prep_args = getattr(platform_cfg, "snpe_dlc_graph_prepare_args", [])
    fb16 = any(
        a == "--float_bitwidth" and str(b) == "16"
        for a, b in pairwise(onnx_args)
    ) or any(
        isinstance(x, str)
        and x.startswith("--float_bitwidth=")
        and x.split("=", 1)[1] == "16"
        for x in onnx_args
    )
    return getattr(platform_cfg, "compress_to_fp16", False) or (
        fb16 and "--use_float_io" in prep_args
    )


def _converted_layout(
    tensor: InputConfig | OutputConfig, new_shape: list[int]
) -> str:
    """Guess the layout of a converted tensor from its configured layout."""
    if tensor.shape is None or any(s == 0 for s in tensor.shape):
        return make_default_layout(new_shape)
    assert tensor.layout is not None
    return guess_new_layout(tensor.layout, tensor.shape, new_shape)


def _output_layout(out: OutputConfig, new_shape: list[int]) -> str:
    """Guess the layout of a converted output from its configured layout.

    When the configured shape does not fit the converted one, the
    output gets the default layout, with a warning.
    """
    try:
        return _converted_layout(out, new_shape)
    except ValueError as e:
        layout = make_default_layout(new_shape)
        logger.warning(
            f"Unable to infer layout for layer '{out.name}': {e}. "
            f"The original shape was `{out.shape}`, which is incompatible "
            f"with the shape of the converted model: `{new_shape}`. "
            f"Changing the layout of the converted model to `{layout}`. "
        )
        return layout


def _archive_input_type(
    inp: InputConfig,
    preprocessing_input_types: dict[str, Literal["raw", "image"]] | None,
) -> Literal["raw", "image"]:
    """Return the input type captured before externalization, or derive it."""
    if (
        preprocessing_input_types is not None
        and inp.name in preprocessing_input_types
    ):
        return preprocessing_input_types[inp.name]
    return default_archive_input_type(is_raw_input=inp.is_raw_input)


def _archive_input_preprocessing(
    inp: InputConfig,
    layout: str,
    input_type: Literal["raw", "image"],
    block: PreprocessingBlock | None,
    orig_nn: NNArchiveConfig | None,
) -> Params:
    """Build the archive preprocessing of an input.

    Without a supplied block, the preprocessing is identity. An image
    input also gets a ``resize_mode`` where the archive supports it.
    """
    if block is None:
        preprocessing_cfg = _default_archive_preprocessing(
            inp, layout, input_type=input_type
        )
    else:
        if input_type == "image":
            block = _adapt_preprocessing_to_layout(block, layout)
        preprocessing_cfg = block.model_dump(mode="json")

    if (
        input_type == "image"
        and "resize_mode" in PreprocessingBlock.model_fields
    ):
        _set_resize_mode(preprocessing_cfg, inp, orig_nn)
    return preprocessing_cfg


def _set_resize_mode(
    preprocessing_cfg: Params,
    inp: InputConfig,
    orig_nn: NNArchiveConfig | None,
) -> None:
    """Set ``resize_mode`` from the calibration config or the old archive.

    The configured resize method wins. Otherwise, a ``resize_mode`` that
    the preprocessing does not set comes from the original archive.
    """
    if inp.calibration.has_resize_method:
        preprocessing_cfg["resize_mode"] = (
            inp.calibration.resize_method.as_nn_archive()
        )
        if preprocessing_cfg["resize_mode"] is None:
            logger.warning(
                f"Input '{inp.name}' uses CENTER_CROP_NO_RESIZE, "
                "which NN Archive resize_mode cannot express. "
                "Exporting null (unspecified);"
            )
        return
    if preprocessing_cfg.get("resize_mode") is not None or not orig_nn:
        return
    original = next(
        (i for i in orig_nn.model.inputs if i.name == inp.name),
        None,
    )
    if original is not None:
        preprocessing_cfg["resize_mode"] = getattr(
            original.preprocessing, "resize_mode", None
        )


def _attach_postprocessor(
    archive: NNArchiveConfig,
    config: Config,
    main_stage_key: str,
    model_name: Path,
) -> None:
    """Point the first head of the archive to the second stage model."""
    if len(config.stages) > 2:
        raise NotImplementedError(
            "Only 2-stage models are supported with NN Archive for now."
        )
    post_stage_key = next(
        key for key in config.stages if key != main_stage_key
    )
    if not archive.model.heads:
        raise ValueError(
            "Multistage NN Archives must specify 1 head in the archive config"
        )
    head = archive.model.heads[0]
    head.metadata.postprocessor_path = f"{post_stage_key}{model_name.suffix}"


def _warn_if_renamed(
    kind: Literal["input", "output"], configured_name: str, converted_name: str
) -> None:
    """Warn when the conversion renamed a tensor."""
    if converted_name != configured_name:
        logger.warning(
            f"Converted model {kind} '{configured_name}' was renamed to "
            f"'{converted_name}'. Using the converted name in the NN Archive."
        )


def modelconverter_config_to_nn(
    config: Config,
    model_name: Path,
    orig_nn: NNArchiveConfig | None,
    preprocessing: dict[str, PreprocessingBlock],
    main_stage_key: str,
    model_path: Path,
    platform: Platform,
    preprocessing_input_types: dict[str, Literal["raw", "image"]]
    | None = None,
) -> NNArchiveConfig:
    """Build the archive config describing a converted model.

    Shapes and data types are taken from the converted model itself,
    layouts are guessed from the original ones, and the precision is
    derived from the platform together with its quantization settings.
    Of the original archive config, the heads are carried over. Input types
    are derived from the effective conversion config, or restored from the
    values captured before preprocessing was externalized. Archive
    preprocessing is identity unless supplied explicitly through
    ``preprocessing``. When supported, ``resize_mode`` uses the configured
    policy, falling back to supplied preprocessing or the input archive.

    Args:
        config: Config the conversion was run with.
        model_name: File name the converted model is stored under
            inside the archive.
        orig_nn: Archive config the conversion started from, or
            ``None`` if it did not start from an archive.
        preprocessing: Preprocessing blocks to attach, keyed by input
            name.
        main_stage_key: Key of the stage holding the main model.
        model_path: Path to the model whose metadata the shapes and
            data types are read from.
        platform: Platform the model was built for.
        preprocessing_input_types: Original input types captured before
            externalized preprocessing cleared the conversion config's
            encodings.

    Returns:
        The archive config for the converted model.

    Raises:
        NotImplementedError: If the config has more than two stages.
        ValueError: If a multi-stage config's archive declares no head.

    """
    model_metadata = get_metadata(model_path)

    cfg = config.stages[main_stage_key]
    platform_cfg = cfg.get_platform_config(platform)

    archive_cfg = {
        "config_version": CONFIG_VERSION,
        "model": {
            "metadata": {
                "name": model_name.stem,
                "path": str(model_name),
                "precision": _archive_precision(platform, platform_cfg).value,
            },
            "inputs": [],
            "outputs": [],
            "heads": orig_nn.model.heads if orig_nn else [],
        },
    }

    input_name_map = _match_tensor_names(
        cfg.inputs, model_metadata.input_shapes, kind="input"
    )
    for inp in cfg.inputs:
        metadata_name = input_name_map[inp.name]
        _warn_if_renamed("input", inp.name, metadata_name)
        new_shape = model_metadata.input_shapes[metadata_name]
        layout = _converted_layout(inp, new_shape)
        dtype = _get_io_dtype(
            platform,
            metadata_name,
            model_metadata,
            platform_cfg,
            mode="input",
        )
        input_type = _archive_input_type(inp, preprocessing_input_types)
        archive_cfg["model"]["inputs"].append(
            {
                "name": metadata_name,
                "shape": new_shape,
                "layout": layout,
                "dtype": dtype,
                "input_type": input_type,
                "preprocessing": _archive_input_preprocessing(
                    inp,
                    layout,
                    input_type,
                    preprocessing.get(inp.name),
                    orig_nn,
                ),
            }
        )
    metadata_output_names = list(model_metadata.output_shapes)
    configured_output_names = [out.name for out in cfg.outputs]
    if len(configured_output_names) != len(metadata_output_names):
        raise ValueError(
            "The converted model has a different number of outputs than the "
            f"conversion config: {metadata_output_names} != "
            f"{configured_output_names}."
        )

    output_name_map = _match_tensor_names(
        cfg.outputs, model_metadata.output_shapes, kind="output"
    )
    renamed_outputs = {
        old_name: new_name
        for old_name, new_name in output_name_map.items()
        if old_name != new_name
    }
    if orig_nn is not None and renamed_outputs:
        archive_cfg["model"]["heads"] = _replace_names(
            orig_nn.model.heads, renamed_outputs
        )
    for out in cfg.outputs:
        metadata_name = output_name_map[out.name]
        _warn_if_renamed("output", out.name, metadata_name)

        new_shape = model_metadata.output_shapes[metadata_name]
        archive_cfg["model"]["outputs"].append(
            {
                "name": metadata_name,
                "shape": new_shape,
                "layout": _output_layout(out, new_shape),
                "dtype": _get_io_dtype(
                    platform,
                    metadata_name,
                    model_metadata,
                    platform_cfg,
                    mode="output",
                ),
            }
        )

    input_names = {inp.name for inp in cfg.inputs}
    unknown_preprocessing = preprocessing.keys() - input_names
    if unknown_preprocessing:
        names = ", ".join(sorted(unknown_preprocessing))
        raise ValueError(f"Preprocessing input(s) not found: {names}")

    archive = NNArchiveConfig(**archive_cfg)
    if len(config.stages) > 1:
        _attach_postprocessor(archive, config, main_stage_key, model_name)
    return archive


def _match_tensor_names(
    configured_tensors: list[InputConfig] | list[OutputConfig],
    converted_shapes: dict[str, list[int]],
    *,
    kind: Literal["input", "output"],
) -> dict[str, str]:
    """Match configured tensors to converted tensors without relying on order.

    Exact names take precedence. Renamed tensors are paired by shape only when
    no other unmatched configured tensor has the same dimensions in a different
    order. A single pair left after those matches is necessarily unambiguous
    even if conversion changed its shape.

    Args:
        configured_tensors: Inputs or outputs from the conversion configuration.
        converted_shapes: Converted tensor shapes keyed by tensor name.
        kind: Tensor kind, used in ambiguity errors.

    Returns:
        A mapping from configured tensor names to converted tensor names.

    Raises:
        ValueError: If renamed tensors cannot be paired unambiguously.

    """
    matches = {
        tensor.name: tensor.name
        for tensor in configured_tensors
        if tensor.name in converted_shapes
    }
    configured_names = {tensor.name for tensor in configured_tensors}
    renamed_shapes = {
        name: shape
        for name, shape in converted_shapes.items()
        if name not in configured_names
    }
    unmatched = [t for t in configured_tensors if t.name not in matches]
    matches |= _match_by_unique_shape(unmatched, renamed_shapes, kind=kind)

    matched = set(matches.values())
    matches |= _match_remaining_pair(
        [t for t in unmatched if t.name not in matches],
        {
            name: shape
            for name, shape in renamed_shapes.items()
            if name not in matched
        },
        kind=kind,
    )
    return matches


def _match_by_unique_shape(
    configured: list[InputConfig] | list[OutputConfig],
    converted_shapes: dict[str, list[int]],
    *,
    kind: Literal["input", "output"],
) -> dict[str, str]:
    """Pair configured and converted tensors that alone have one shape.

    Raises:
        ValueError: If another configured tensor has the same
            dimensions in a different order.
    """
    configured_by_shape: dict[
        tuple[int, ...], list[InputConfig | OutputConfig]
    ] = {}
    for tensor in configured:
        if tensor.shape is not None:
            configured_by_shape.setdefault(tuple(tensor.shape), []).append(
                tensor
            )

    converted_by_shape: dict[tuple[int, ...], list[str]] = {}
    for name, shape in converted_shapes.items():
        converted_by_shape.setdefault(tuple(shape), []).append(name)

    matches = {}
    for shape, tensors in configured_by_shape.items():
        converted_names = converted_by_shape.get(shape, [])
        if not len(tensors) == len(converted_names) == 1:
            continue
        permutation_candidates = [
            tensor.name
            for tensor in configured
            if tensor.shape is not None
            and sorted(tensor.shape) == sorted(shape)
        ]
        if len(permutation_candidates) > 1:
            raise ValueError(
                f"Unable to unambiguously match renamed model {kind}s by "
                f"shape: converted tensor '{converted_names[0]}' with shape "
                f"{list(shape)} could correspond to any of "
                f"{permutation_candidates} after an axis permutation."
            )
        matches[tensors[0].name] = converted_names[0]
    return matches


def _match_remaining_pair(
    configured: list[InputConfig] | list[OutputConfig],
    converted_shapes: dict[str, list[int]],
    *,
    kind: Literal["input", "output"],
) -> dict[str, str]:
    """Pair the one configured tensor left with the one converted tensor left.

    Raises:
        ValueError: If more than one tensor is left on a side, or a
            tensor is left on one side only.
    """
    if len(configured) == len(converted_shapes) == 1:
        return {configured[0].name: next(iter(converted_shapes))}
    if configured or converted_shapes:
        configured_shapes = {
            tensor.name: tensor.shape for tensor in configured
        }
        raise ValueError(
            f"Unable to unambiguously match renamed model {kind}s by shape: "
            f"configured={configured_shapes}, converted={converted_shapes}."
        )
    return {}


def _replace_names(value: object, name_map: dict[str, str]) -> object:
    """Recursively replace exact tensor names in archive data, keys included."""
    if isinstance(value, str):
        return name_map.get(value, value)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        return {
            _replace_names(key, name_map): _replace_names(item, name_map)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_names(item, name_map) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_names(item, name_map) for item in value)
    return value


def default_archive_input_type(
    *, is_raw_input: bool
) -> Literal["raw", "image"]:
    """Map a modelconverter input to its default NN Archive input type."""
    if is_raw_input:
        return "raw"
    return "image"


def make_dai_type(
    encoding: Encoding, data_type: DataType, layout: str | None
) -> str:
    """Build a DepthAI image type for an archive preprocessing block."""
    if encoding == Encoding.NONE:
        raise ValueError(
            "Cannot build a DepthAI image type for Encoding.NONE."
        )

    if encoding == Encoding.GRAY:
        channel_format = "F16" if data_type == DataType.FLOAT16 else "8"
        return f"{encoding.value}{channel_format}"

    channel_format = "F16F16F16" if data_type == DataType.FLOAT16 else "888"
    storage = "i" if is_interleaved_image_layout(layout) else "p"
    return f"{encoding.value}{channel_format}{storage}"


def _adapt_preprocessing_to_layout(
    block: PreprocessingBlock, layout: str
) -> PreprocessingBlock:
    """Return externalized image preprocessing for the converted layout."""
    dai_type = block.dai_type
    if dai_type is not None and dai_type.startswith(("RGB", "BGR")):
        if dai_type.endswith(("i", "p")):
            dai_type = dai_type[:-1]
        dai_type += "i" if is_interleaved_image_layout(layout) else "p"

    return block.model_copy(
        update={
            "interleaved_to_planar": is_interleaved_image_layout(layout),
            "dai_type": dai_type,
        }
    )


def _default_archive_preprocessing(
    inp: InputConfig,
    layout: str,
    *,
    input_type: Literal["raw", "image"],
) -> Params:
    inp.validate_input_contract()
    if input_type == "raw":
        return {
            "mean": None,
            "scale": None,
            "reverse_channels": None,
            "interleaved_to_planar": None,
            "dai_type": None,
        }

    dai_type = make_dai_type(inp.encoding.to, inp.data_type, layout)

    return {
        "mean": (
            broadcast_preprocessing_values([0], inp.channel_count)
            if inp.mean_values
            else None
        ),
        "scale": (
            broadcast_preprocessing_values([1], inp.channel_count)
            if inp.scale_values
            else None
        ),
        "reverse_channels": inp.encoding.to == Encoding.RGB,
        "interleaved_to_planar": is_interleaved_image_layout(layout),
        "dai_type": dai_type,
    }


def archive_from_model(model_path: Path) -> NNArchiveConfig:
    """Build a bare archive config out of a model file alone.

    The inputs and outputs come from the model's own metadata. Inputs with a
    supported image shape and layout are declared as images; all others are
    raw tensors. No preprocessing or heads are declared. This is what packing
    an unconverted model into an archive starts from.

    Args:
        model_path: Path to the model file to describe.

    Returns:
        The archive config describing the model.

    """
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
        layout = make_default_layout(shape)
        input_type = "image" if is_image_input_shape(shape, layout) else "raw"
        archive_cfg["model"]["inputs"].append(
            {
                "name": name,
                "shape": shape,
                "layout": layout,
                "dtype": metadata.input_dtypes[name].value,
                "input_type": input_type,
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
    platform: Platform,
    cfg: Config,
    main_stage: str,
    out_models: list[Path],
    output_path: Path,
    archive_cfg: NNArchiveConfig | None,
    preprocessing: dict[str, PreprocessingBlock],
    inference_model_path: Path,
    preprocessing_input_types: dict[str, Literal["raw", "image"]]
    | None = None,
) -> Path:
    """Pack the converted models into an NN Archive.

    The archive holds the model files together with the generated
    config and the build info, and its name is suffixed with the
    platform it was built for.

    Args:
        platform: Platform the models were built for.
        cfg: Config the conversion was run with.
        main_stage: Key of the stage holding the main model.
        out_models: Converted model files to put into the archive.
        output_path: Directory the archive is written to, and where
            ``buildinfo.json`` is picked up from.
        archive_cfg: Archive config the conversion started from, or
            ``None`` if it did not start from an archive.
        preprocessing: Preprocessing blocks to attach, keyed by input
            name.
        inference_model_path: Path to the model whose metadata the
            shapes and data types are read from.
        preprocessing_input_types: Original input types captured before
            externalizing preprocessing from the conversion config.

    Returns:
        Path to the created archive.

    """
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
        platform,
        preprocessing_input_types=preprocessing_input_types,
    )
    generator = ArchiveGenerator(
        archive_name=f"{cfg.name}.{platform.value.lower()}",
        save_path=str(output_path),
        cfg_dict=nn_archive.model_dump(),
        executables_paths=[
            *out_models,
            output_path / "buildinfo.json",
        ],
    )
    return generator.make_archive()


def _get_io_dtype(
    platform: Platform,
    name: str,
    metadata: Metadata,
    cfg: PlatformConfig,
    *,
    mode: Literal["input", "output"],
) -> str:
    dtypes = (
        metadata.input_dtypes if mode == "input" else metadata.output_dtypes
    )
    if platform not in {Platform.RVC2, Platform.RVC3}:
        return dtypes[name].as_nn_archive_dtype()
    assert isinstance(cfg, BlobBaseConfig)
    blob_dtype = _compile_tool_dtype(cfg.compile_tool_args, name, mode=mode)
    if blob_dtype is None:
        return dtypes[name].as_nn_archive_dtype()
    return DataType.from_ir_ie_dtype(blob_dtype).as_nn_archive_dtype()


def _compile_tool_dtype(
    args: list[str], name: str, *, mode: Literal["input", "output"]
) -> str | None:
    """Read the precision that ``compile_tool`` arguments set for a tensor.

    ``-iop "<name1>:<dtype1>,<name2>:<dtype2>"`` sets it per tensor, and
    wins over ``-ip`` and ``-op``, which set it for every input or every
    output. A name that ``-iop`` does not list leaves its whole value,
    which `DataType.from_ir_ie_dtype` refuses.

    Returns:
        The precision in upper case, or ``None`` when no argument sets
        it.
    """
    if "-iop" in args:
        value = args[args.index("-iop") + 1]
        for item in value.split(","):
            tensor_name, dtype = item.strip().split(":")
            if tensor_name == name:
                return dtype.upper()
        return value.upper()
    flag = "-ip" if mode == "input" else "-op"
    if flag in args:
        return args[args.index(flag) + 1].upper()
    return None
