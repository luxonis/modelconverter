import warnings
from itertools import chain
from pathlib import Path
from typing import Annotated, Literal

import onnx
from loguru import logger
from luxonis_ml.typing import (
    BaseModelExtraForbid,
    Params,
    ParamValue,
    PathType,
)
from luxonis_ml.utils import LuxonisConfig
from onnx import TypeProto
from pydantic import (
    Field,
    PositiveInt,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from modelconverter.utils.calibration_data import download_calibration_data
from modelconverter.utils.constants import MISC_DIR, MODELS_DIR
from modelconverter.utils.encodings import parse_encodings
from modelconverter.utils.filesystem_utils import resolve_path
from modelconverter.utils.layout import make_default_layout
from modelconverter.utils.metadata import Metadata, get_metadata
from modelconverter.utils.onnx_compatibility import (
    has_external_data,
    save_onnx_model,
)
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    Platform,
    PotDevice,
    QuantizationMode,
    ResizeMethod,
)

NAMED_VALUES = {
    "imagenet": {
        "mean": [123.675, 116.28, 103.53],
        "scale": [58.395, 57.12, 57.375],
    },
}


class LinkCalibrationConfig(BaseModelExtraForbid):
    stage: str
    output: str | None = None
    script: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.output is None and self.script is None:
            raise ValueError(
                "Either `output` or `script` must be provided for calibration."
            )
        return self

    @field_validator("script", mode="after")
    @staticmethod
    def _download_calibration_script(script: str | None) -> str | None:
        if script is None:
            return None
        if script.endswith(".py"):
            script_path = resolve_path(script, MODELS_DIR)
            script = script_path.read_text()
        return script


class ImageCalibrationConfig(BaseModelExtraForbid):
    path: Path
    max_images: int = -1
    resize_method: ResizeMethod = ResizeMethod.RESIZE

    @field_validator("path", mode="before")
    @staticmethod
    def _download_calibration_data(value: ParamValue) -> Path | None:
        if value is None:
            return None
        return download_calibration_data(str(value))


class RandomCalibrationConfig(BaseModelExtraForbid):
    max_images: int = 20
    min_value: float = 0.0
    max_value: float = 255.0
    mean: float = 127.5
    std: float = 35.0
    data_type: DataType = DataType.FLOAT32


class OutputConfig(BaseModelExtraForbid):
    name: str
    shape: list[int] | None = None
    layout: str | None = None
    data_type: DataType = DataType.FLOAT32

    @model_validator(mode="before")
    @classmethod
    def _make_default_layout(cls, data: Params) -> Params:
        shape = data.get("shape")
        layout = data.get("layout")
        if shape is None and layout is not None:
            raise ValueError("`layout` cannot be provided without `shape`.")
        if shape is None:
            return data
        if layout is None:
            layout = make_default_layout(_as_shape(shape, "shape"))
        elif not isinstance(layout, str):
            raise TypeError("`layout` must be a string.")
        data["layout"] = layout.upper()
        return data

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        if self.shape is None:
            return self
        assert self.layout is not None
        if len(self.layout) != len(self.shape):
            raise ValueError(
                f"Length of `layout` ({len(self.layout)}) must match "
                f"length of `shape` ({len(self.shape)})."
            )
        return self


class EncodingConfig(BaseModelExtraForbid):
    from_: Annotated[
        Encoding, Field(alias="from", serialization_alias="from")
    ] = Encoding.RGB
    to: Encoding = Encoding.BGR


class InputConfig(OutputConfig):
    calibration: (
        ImageCalibrationConfig
        | RandomCalibrationConfig
        | LinkCalibrationConfig
    ) = RandomCalibrationConfig()
    scale_values: Annotated[list[float], Field(min_length=1)] | None = None
    mean_values: Annotated[list[float], Field(min_length=1)] | None = None
    frozen_value: list[int | float] | None = None
    encoding: EncodingConfig = EncodingConfig()

    @property
    def encoding_mismatch(self) -> bool:
        return self.encoding.from_ != self.encoding.to

    @property
    def is_color_input(self) -> bool:
        return self.encoding.from_ in {Encoding.RGB, Encoding.BGR}

    @property
    def is_raw_input(self) -> bool:
        return (
            self.encoding.from_ == Encoding.NONE
            and self.encoding.to == Encoding.NONE
        )

    @model_validator(mode="after")
    def _validate_grayscale_inputs(self) -> Self:
        if self.layout is None:
            return self

        if "C" not in self.layout:
            return self

        assert self.shape is not None

        channels = self.shape[self.layout.index("C")]
        if channels == 1:
            logger.info("Detected grayscale input. Setting encoding to GRAY.")
            self.encoding.from_ = self.encoding.to = Encoding.GRAY

        return self

    @model_validator(mode="after")
    def _validate_dynamic_batch_size(self) -> Self:
        if self.shape is not None and self.shape[0] == 0:
            logger.info(
                "Detected dynamic batch size (the first element "
                "of the shape is set to 0). Setting batch size to 1. "
            )
            self.shape[0] = 1
        return self

    @model_validator(mode="before")
    @classmethod
    def _validate_encoding(cls, data: Params) -> Params:
        encoding = data.get("encoding")
        if encoding is None or encoding == {}:
            data["encoding"] = {"from": "RGB", "to": "BGR"}
            return data
        if isinstance(encoding, str):
            data["encoding"] = {"from": encoding, "to": encoding}
        if isinstance(encoding, dict) and (
            ("from" in encoding and encoding["from"] == "GRAY")
            or ("to" in encoding and encoding["to"] == "GRAY")
        ):
            data["encoding"] = {"from": "GRAY", "to": "GRAY"}
        return data

    @model_validator(mode="before")
    @classmethod
    def _random_calibration(cls, data: Params) -> Params:
        if data.get("calibration") in ["random", None]:
            # An empty mapping builds the config from its defaults.
            data["calibration"] = {}
        return data

    @field_validator("scale_values", mode="before")
    @staticmethod
    def _parse_scale_values(value: ParamValue) -> ParamValue:
        """Parses the scale_values from the config."""
        return InputConfig._parse_values("scale", value)

    @field_validator("mean_values", mode="before")
    @staticmethod
    def _parse_mean_values(value: ParamValue) -> ParamValue:
        """Parses the mean_values from the config."""
        return InputConfig._parse_values("mean", value)

    @staticmethod
    def _parse_values(
        values_type: Literal["mean", "scale"], value: ParamValue
    ) -> ParamValue:
        """Resolves named values from the config."""
        if value is None:
            return None
        if isinstance(value, str) and value in NAMED_VALUES:
            return NAMED_VALUES[value][values_type]
        if isinstance(value, float | int):
            return [value, value, value]
        return value

    def requires_onnx_input_modification(
        self, *, reverse_only: bool = False
    ) -> bool:
        if self.encoding_mismatch:
            return True
        if reverse_only:
            return False
        return (
            self.mean_values is not None
            and any(v != 0 for v in self.mean_values)
        ) or (
            self.scale_values is not None
            and any(v != 1 for v in self.scale_values)
        )


class PlatformConfig(BaseModelExtraForbid):
    disable_calibration: bool = False


class HailoConfig(PlatformConfig):
    force_onnx_names: bool = True
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] = 2
    compression_level: Literal[0, 1, 2, 3, 4, 5] = 2
    batch_size: int = 8
    disable_compilation: bool = False
    alls: list[str] = []
    hw_arch: Literal[
        "hailo8", "hailo8l", "hailo8r", "hailo10h", "hailo15h", "hailo15m"
    ] = "hailo8"


class BlobBaseConfig(PlatformConfig):
    mo_args: list[str] = []
    compile_tool_args: list[str] = []
    compress_to_fp16: bool = True


class RVC2Config(BlobBaseConfig):
    number_of_shaves: int = 8
    superblob: bool = True
    n_workers: PositiveInt | None = None

    @model_validator(mode="after")
    def _validate_superblob(self) -> Self:
        if self.superblob and self.number_of_shaves != 8:
            logger.warning("Changing number_of_shaves to 8 for superblob.")
            self.number_of_shaves = 8

        return self


class RVC3Config(BlobBaseConfig):
    pot_target_device: PotDevice = PotDevice.VPU


class QuantizationOverridesItem(BaseModelExtraForbid):
    bitwidth: Annotated[int, Literal[4, 8, 16, 32]] | None = None
    is_symmetric: bool | None = None
    dtype: Literal["int", "float"] | None = None
    max: float | None = None
    min: float | None = None
    offset: int | None = None
    scale: float | None = None

    @field_serializer("is_symmetric", when_used="json")
    @staticmethod
    def serialize_is_symmetric(value: bool | None) -> str | None:
        if value is None:
            return None
        return str(value)


class Encodings(BaseModelExtraForbid):
    activation_encodings: dict[str, list[QuantizationOverridesItem]]
    param_encodings: dict[str, list[QuantizationOverridesItem]]


class RVC4Config(PlatformConfig):
    snpe_onnx_to_dlc_args: list[str] = []
    snpe_dlc_quant_args: list[str] = []
    snpe_dlc_graph_prepare_args: list[str] = []
    keep_raw_images: bool = False
    use_per_channel_quantization: bool = True
    use_per_row_quantization: bool = False
    optimization_level: Literal[1, 2, 3] = 2
    quantization_mode: QuantizationMode = QuantizationMode.INT8_STD
    htp_socs: list[
        Literal["sm8350", "sm8450", "sm8550", "sm8650", "qcs6490", "qcs8550"]
    ] = ["sm8550"]
    encodings: Encodings | None = None

    @model_validator(mode="after")
    def validate_quantization_overrides(self) -> Self:
        if "--quantization_overrides" in self.snpe_onnx_to_dlc_args:
            if self.encodings:
                raise ValueError(
                    "Cannot specify both `--quantization_overrides`"
                    "in `rvc4.snpe_onnx_to_dlc_args` and "
                    "`rvc4.encodings` at the same time."
                )
            qo_index = self.snpe_onnx_to_dlc_args.index(
                "--quantization_overrides"
            )
            self.snpe_onnx_to_dlc_args.pop(qo_index)
            encodings_json = self.snpe_onnx_to_dlc_args.pop(qo_index)
            with open(encodings_json) as f:
                self.encodings = parse_encodings(f.read())
        return self

    @field_validator("encodings", mode="before")
    @staticmethod
    def validate_encodings(value: ParamValue | Encodings) -> Encodings | None:
        if value is None:
            return None

        if isinstance(value, str):
            if value.lstrip().startswith("{"):
                return parse_encodings(value)
            value_path = resolve_path(value, MISC_DIR)
            return parse_encodings(value_path.read_text())

        return parse_encodings(value)

    @model_validator(mode="after")
    def _validate_fp16(self) -> Self:
        if self.quantization_mode != QuantizationMode.FP16_STD:
            return self
        self.disable_calibration = True
        return self


class ONNXOptimizationsConfig(BaseModelExtraForbid):
    fuse_add_mul_to_bn: bool = True
    fuse_comb_add_mul_to_conv: bool = True
    fuse_single_add_mul_to_conv: bool = True
    fuse_split_concat_to_conv: bool = True
    substitute_sub_with_add: bool = True
    substitute_div_with_mul: bool = True

    def all_disabled(self) -> bool:
        return not any(
            [
                self.fuse_add_mul_to_bn,
                self.fuse_comb_add_mul_to_conv,
                self.fuse_single_add_mul_to_conv,
                self.fuse_split_concat_to_conv,
                self.substitute_sub_with_add,
                self.substitute_div_with_mul,
            ]
        )


class SingleStageConfig(BaseModelExtraForbid):
    input_model: Path
    input_bin: Path | None = None
    input_file_type: InputFileType

    inputs: Annotated[list[InputConfig], Field(min_length=1)] = []
    outputs: Annotated[list[OutputConfig], Field(min_length=1)] = []

    keep_intermediate_outputs: bool = True
    onnx_simplification: Literal["onnxsim", "onnxslim", False] = "onnxsim"
    onnx_optimizations: ONNXOptimizationsConfig = ONNXOptimizationsConfig()
    output_remote_url: str | None = None
    intermediate_outputs_remote_url: str | None = None
    put_file_plugin: str | None = None

    hailo: HailoConfig = HailoConfig()
    rvc2: RVC2Config = RVC2Config()
    rvc3: RVC3Config = RVC3Config()
    rvc4: RVC4Config = RVC4Config()

    @model_validator(mode="before")
    @classmethod
    def validate_onnx_simplification(cls, data: Params) -> Params:
        if data.pop("disable_onnx_simplification", False):
            if "onnx_simplification" in data:
                raise ValueError(
                    "Cannot specify both `disable_onnx_simplification` "
                    "and `onnx_simplification`."
                )

            warnings.warn(
                "`disable_onnx_simplification` is deprecated. Please use "
                "`onnx_simplification` set to `False` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["onnx_simplification"] = False
        return data

    def get_platform_config(self, platform: Platform) -> PlatformConfig:
        """Returns the platform configuration for the given platform."""
        if platform == Platform.HAILO:
            return self.hailo
        if platform == Platform.RVC2:
            return self.rvc2
        if platform == Platform.RVC3:
            return self.rvc3
        if platform == Platform.RVC4:  # pragma: no branch
            return self.rvc4

    @model_validator(mode="before")
    @classmethod
    def validate_onnx_optimizations(cls, data: Params) -> Params:
        if "onnx_optimizations" not in data:
            return data
        optimizations = data["onnx_optimizations"]
        if optimizations in ["all", True]:
            data["onnx_optimizations"] = {}
        elif optimizations in ["none", None, False]:
            data["onnx_optimizations"] = dict.fromkeys(
                ONNXOptimizationsConfig.model_fields, False
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_disable_onnx_optimizations(cls, data: Params) -> Params:
        if "disable_onnx_optimizations" not in data:
            return data
        if data.pop("disable_onnx_optimizations", False):
            warnings.warn(
                "`disable_onnx_optimizations` is deprecated. Please use "
                "`onnx_optimizations: false` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "onnx_optimizations" in data:
                raise ValueError(
                    "Cannot specify both `disable_onnx_optimizations` and "
                    "`onnx_optimizations` at the same time."
                )
            data["onnx_optimizations"] = dict.fromkeys(
                ONNXOptimizationsConfig.model_fields, False
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, data: Params) -> Params:
        mean_values = data.pop("mean_values", None)
        scale_values = data.pop("scale_values", None)
        encoding = data.pop("encoding", {})
        data_type = data.pop("data_type", None)
        shape = data.pop("shape", None)
        layout = data.pop("layout", None)
        top_level_calibration = data.pop("calibration", {})

        model_path = Path(_as_path_type(data["input_model"], "input_model"))

        input_file_type = InputFileType.from_path(model_path)
        data["input_file_type"] = input_file_type.value
        if input_file_type == InputFileType.PYTORCH:
            logger.info(
                "Detected PyTorch model. Only YOLO models are supported."
            )
            raw_shape = data.pop("yolo_input_shape", [640, 640])
            if isinstance(raw_shape, str):
                input_shape = (
                    [int(size) for size in raw_shape.split(" ")]
                    if " " in raw_shape
                    else [int(raw_shape)] * 2
                )
            else:
                logger.warning(
                    "yolo_input_shape is not provided. Using default shape [640, 640]."
                )
                input_shape = _as_shape(raw_shape, "yolo_input_shape")
            input_shapes = {"images": input_shape[::-1]}
            input_dtypes = {"images": DataType.FLOAT32}
            output_shapes = {"dummy": [0]}
            output_dtypes = {"dummy": DataType.FLOAT32}

            metadata = Metadata(
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
            )
        else:
            metadata = get_metadata(model_path)

        inputs: list[Params] = _as_entries(data.get("inputs"), "inputs")
        if not inputs:
            inputs = [{"name": name} for name in metadata.input_shapes]
        outputs: list[Params] = _as_entries(data.get("outputs"), "outputs")
        if not outputs:
            outputs = [{"name": name} for name in metadata.output_shapes]

        for inp in inputs:
            if "name" not in inp:
                raise ValueError(
                    f"Unable to determine name for input: `{inp}`."
                )
            inp_name = str(inp["name"])
            if inp_name in metadata.input_shapes:
                onnx_shape, onnx_dtype = (
                    metadata.input_shapes[inp_name],
                    metadata.input_dtypes[inp_name],
                )
            else:
                onnx_shape, onnx_dtype = _get_onnx_inter_info(
                    model_path, inp_name
                )
                logger.warning(
                    f"Input `{inp_name}` is not present in inputs of the ONNX model. "
                    f"Assuming it is an intermediate node."
                )
            inp["shape"] = inp.get("shape") or shape or onnx_shape
            inp["layout"] = inp.get("layout") or layout
            inp["data_type"] = (
                inp.get("data_type") or data_type or _dtype_name(onnx_dtype)
            )
            inp["encoding"] = inp.get("encoding") or encoding
            inp["mean_values"] = (
                inp.get("mean_values")
                if inp.get("mean_values") is not None
                else mean_values
            )
            inp["scale_values"] = (
                inp.get("scale_values")
                if inp.get("scale_values") is not None
                else scale_values
            )

            inp_calibration = inp.get("calibration", {})
            if not inp_calibration and not top_level_calibration:
                inp["calibration"] = None
            elif top_level_calibration == "random":
                inp["calibration"] = "random"
            else:
                inp["calibration"] = {
                    **_as_dict(top_level_calibration, "calibration"),
                    **_as_dict(inp_calibration, "calibration"),
                }

        for out in outputs:
            out_name = str(out["name"])
            if (
                out_name not in metadata.output_shapes
                and out.get("data_type") is None
                and out.get("shape") is None
            ):
                onnx_shape, onnx_dtype = _get_onnx_inter_info(
                    model_path, out_name
                )
            elif out_name in metadata.output_shapes:
                onnx_shape, onnx_dtype = (
                    metadata.output_shapes[out_name],
                    metadata.output_dtypes[out_name],
                )
            else:
                onnx_shape, onnx_dtype = None, None
            out["shape"] = out.get("shape") or onnx_shape
            out["data_type"] = out.get("data_type") or _dtype_name(onnx_dtype)

        data["inputs"] = inputs
        data["outputs"] = outputs

        disable_calibration = data.pop("disable_calibration", None)
        if disable_calibration is None:
            return data

        for platform in ["hailo", "rvc2", "rvc3", "rvc4"]:
            platform_data = _as_dict(data.setdefault(platform, {}), platform)
            platform_data["disable_calibration"] = disable_calibration
        return data

    @model_validator(mode="before")
    @classmethod
    def _download_input_model(cls, value: Params) -> Params:
        if "input_model" not in value:
            raise ValueError("`input_model` must be provided.")
        input_model = _as_path_type(value["input_model"], "input_model")
        input_file_type = InputFileType.from_path(input_model)
        if input_file_type == InputFileType.IR:
            bin_path, xml_path = _extract_bin_xml_from_ir(input_model)
            value["input_bin"] = str(bin_path)
            value["input_model"] = str(xml_path)
        else:
            value["input_model"] = str(
                resolve_path(str(input_model), MODELS_DIR)
            )
        return value


# TODO: Output remote url
class Config(LuxonisConfig):
    stages: Annotated[dict[str, SingleStageConfig], Field(min_length=1)]
    name: str
    rich_logging: bool = True

    def get_stage_config(self, stage: str | None) -> SingleStageConfig:
        if stage is None:
            if len(self.stages) == 1:
                return next(iter(self.stages.values()))
            raise ValueError("Multiple stages found. Please specify a stage.")
        return self.stages[stage]

    @model_validator(mode="before")
    @classmethod
    def _validate_name(cls, data: Params) -> Params:
        if data.get("name") is None:
            stages = _as_dict(data["stages"], "stages")
            data["name"] = "-".join(stages.keys())
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_stages(cls, data: Params) -> Params:
        if "stages" not in data:
            name = data.pop("name", "default_stage")
            if not isinstance(name, str):
                raise TypeError("`name` must be a string.")
            rich_logging = data.pop("rich_logging", True)
            return {
                "name": name,
                "rich_logging": rich_logging,
                "stages": {name: data},
            }

        extra: Params = {}
        for key in list(data.keys()):
            if key not in cls.model_fields:
                extra[key] = data.pop(key)
        stages = _as_dict(data["stages"], "stages")
        for stage_name, stage in stages.items():
            stage_data = _as_dict(stage, f"stages.{stage_name}")
            for key, value in extra.items():
                if key not in stage_data:
                    stage_data[key] = value
        return data

    @model_validator(mode="after")
    def _validate_single_stage_name(self) -> Self:
        """Changes the default 'default_stage' name to the name of the
        input model."""
        if len(self.stages) == 1 and "default_stage" in self.stages:
            stage = next(iter(self.stages.values()))
            model_name = stage.input_model.stem
            self.stages = {model_name: stage}
            self.name = model_name
        return self


def _dtype_name(dtype: DataType | None) -> str | None:
    return None if dtype is None else dtype.value


def _as_dict(value: ParamValue, name: str) -> Params:
    if not isinstance(value, dict):
        raise TypeError(f"`{name}` must be a dictionary.")
    return value


def _as_entries(value: ParamValue, name: str) -> list[Params]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"`{name}` must be a list of dictionaries.")
    entries = [entry for entry in value if isinstance(entry, dict)]
    if len(entries) != len(value):
        raise TypeError(f"`{name}` must be a list of dictionaries.")
    return entries


def _as_shape(value: ParamValue, name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"`{name}` must be a list of integers.")
    dims = [dim for dim in value if isinstance(dim, int)]
    if len(dims) != len(value):
        raise TypeError(f"`{name}` must be a list of integers.")
    return dims


def _as_path_type(value: ParamValue, name: str) -> PathType:
    """Narrows a raw configuration value to something C{Path} accepts.

    Returned as it arrived rather than as a C{Path}: a remote location
    is carried in these fields as a string, and C{Path} would fold the
    C{//} of its protocol away.
    """
    if not isinstance(value, PathType):
        raise TypeError(f"`{name}` must be a string or a path.")
    return value


def _extract_bin_xml_from_ir(ir_path: ParamValue | Path) -> tuple[Path, Path]:
    """Extracts the corresponding second path from a single IR path.

    We assume that the base filename matches between the .bin and .xml
    file. Otherwise, an error will be thrown.

    @type ir_path: ParamValue | Path
    @param ir_path: The path of either the C{.bin} or the C{.xml} file.
        It arrives unvalidated, straight out of the configuration, so it
        is only a path once the check below has run.
    @rtype: tuple[Path, Path]
    @return: The C{.bin} path and the C{.xml} path.
    """
    if not isinstance(ir_path, PathType):
        raise TypeError("`input_path` must be str or Path.")
    path = Path(ir_path)

    if path.suffix == ".bin":
        bin_path = str(path)
        xml_path = str(path.with_suffix(".xml"))
    elif path.suffix == ".xml":
        xml_path = str(path)
        bin_path = str(path.with_suffix(".bin"))
    else:
        raise ValueError(
            "`ir_path` is invalid: does not have .bin or .xml extension."
        )

    # fix any remote path corruption from pathlib
    bin_path = bin_path.replace(":/", "://")
    xml_path = xml_path.replace(":/", "://")

    try:
        resolved_bin = resolve_path(bin_path, MODELS_DIR)
    except Exception as e:
        raise ValueError(
            f"`bin_path` {bin_path} was not found. "
            "Please ensure that your xml and bin file have matching file basenames "
            "and are located in the same directory."
        ) from e
    try:
        resolved_xml = resolve_path(xml_path, MODELS_DIR)
    except Exception as e:
        raise ValueError(
            f"`xml_path` {xml_path} was not found. "
            "Please ensure that your xml and bin file have matching file basenames and "
            "are located in the same directory."
        ) from e

    return resolved_bin, resolved_xml


def _get_onnx_node_info(
    model_path: Path, node_name: str
) -> tuple[list[int], DataType]:
    onnx_model = onnx.load(str(model_path))
    graph = onnx_model.graph

    node = next((n for n in graph.node if n.name == node_name), None)
    if node is None:
        raise NameError(f"Node '{node_name}' not found in the ONNX model.")

    output_value_info = next(
        (info for info in graph.value_info if info.name == node.output[0]),
        None,
    )

    if output_value_info is None:
        raise ValueError(
            f"Output value info for node '{node_name}' not found."
        )

    shape = _get_static_onnx_shape(
        output_value_info.type.tensor_type, f"node '{node_name}'"
    )
    data_type = output_value_info.type.tensor_type.elem_type

    return shape, DataType.from_onnx_dtype(data_type)


def _get_onnx_tensor_info(
    model_path: PathType, tensor_name: str
) -> tuple[list[int], DataType]:
    model = onnx.load(str(model_path))

    def extract_tensor_info(
        tensor_type: TypeProto.Tensor,
    ) -> tuple[list[int], DataType]:
        shape = _get_static_onnx_shape(tensor_type, f"tensor '{tensor_name}'")
        return shape, DataType.from_onnx_dtype(tensor_type.elem_type)

    for tensor in chain(model.graph.input, model.graph.output):
        if tensor.name == tensor_name:
            return extract_tensor_info(tensor.type.tensor_type)

    for node in model.graph.node:
        for tensor in chain(node.input, node.output):
            if tensor == tensor_name:
                for value_info in model.graph.value_info:
                    if value_info.name == tensor_name:
                        return extract_tensor_info(value_info.type.tensor_type)
                raise ValueError(
                    f"Tensor '{tensor_name}' does not have shape/type information."
                )

    raise NameError(f"Tensor '{tensor_name}' not found in the ONNX model.")


def _get_static_onnx_shape(
    tensor_type: TypeProto.Tensor, tensor_name: str
) -> list[int]:
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            raise ValueError(
                "Dynamic shapes are not supported. "
                f"Shape of {tensor_name} is {[d.dim_value for d in tensor_type.shape.dim]}."
            )
    return shape


def _get_onnx_inter_info(
    model_path: Path, name: str
) -> tuple[list[int] | None, DataType | None]:
    try:
        logger.info(
            f"Attempting to find shape and data type for tensor '{name}'."
        )
        shape, data_type = _get_onnx_tensor_info(model_path, name)
    except (NameError, ValueError) as e:
        logger.warning(str(e))
        logger.info(
            f"Attempting to find shape and data type for node '{name}'."
        )
        try:
            shape, data_type = _get_onnx_node_info(model_path, name)
        except (NameError, ValueError) as e:
            logger.warning(str(e))
            shape, data_type = None, None
    if shape is None or data_type is None:
        logger.warning(
            f"Tensor or node '{name}' not found or does not have shape/type information. "
            "Proceeding without shape and data type information."
        )
    else:
        logger.info(
            f"Found shape and data type for '{name}': {shape}, {data_type.name}"
        )
    return shape, data_type


def generate_renamed_onnx(
    onnx_path: PathType,
    rename_dict: dict[str, str],
    output_path: PathType,
) -> None:
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    model = onnx.load(str(onnx_path))
    model_has_external_data = has_external_data(onnx_path)

    for node in model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in rename_dict:
                node.input[i] = rename_dict[input_name]

        for i, output_name in enumerate(node.output):
            if output_name in rename_dict:
                node.output[i] = rename_dict[output_name]

    save_onnx_model(
        model,
        output_path,
        save_as_external_data=model_has_external_data,
        location=f"{output_path.name}_data",
    )
