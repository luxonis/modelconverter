from itertools import chain
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import onnx
from loguru import logger
from luxonis_ml.typing import PathType
from luxonis_ml.utils import LuxonisConfig
from onnx import TypeProto
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from modelconverter.utils.calibration_data import download_calibration_data
from modelconverter.utils.constants import MODELS_DIR
from modelconverter.utils.filesystem_utils import resolve_path
from modelconverter.utils.layout import make_default_layout
from modelconverter.utils.metadata import Metadata, get_metadata
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    PotDevice,
    ResizeMethod,
    Target,
)

NAMED_VALUES = {
    "imagenet": {
        "mean": [123.675, 116.28, 103.53],
        "scale": [58.395, 57.12, 57.375],
    },
}


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LinkCalibrationConfig(CustomBaseModel):
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
    def _download_calibration_script(script: Any) -> Path | None:
        if script is None:
            return None
        if script.endswith(".py"):
            script_path = resolve_path(script, MODELS_DIR)
            script = script_path.read_text()
        return script


class ImageCalibrationConfig(CustomBaseModel):
    path: Path
    max_images: int = -1
    resize_method: ResizeMethod = ResizeMethod.RESIZE

    @field_validator("path", mode="before")
    @staticmethod
    def _download_calibration_data(value: Any) -> Path | None:
        if value is None:
            return None
        return download_calibration_data(str(value))


class RandomCalibrationConfig(CustomBaseModel):
    max_images: int = 20
    min_value: float = 0.0
    max_value: float = 255.0
    mean: float = 127.5
    std: float = 35.0
    data_type: DataType = DataType.FLOAT32


class OutputConfig(CustomBaseModel):
    name: str
    shape: list[int] | None = None
    layout: str | None = None
    data_type: DataType = DataType.FLOAT32

    @model_validator(mode="before")
    @classmethod
    def _make_default_layout(cls, data: dict[str, Any]) -> dict[str, Any]:
        shape = data.get("shape")
        layout = data.get("layout")
        if shape is None and layout is not None:
            raise ValueError("`layout` cannot be provided without `shape`.")
        if shape is None:
            return data
        if layout is None:
            layout = make_default_layout(shape)
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


class EncodingConfig(CustomBaseModel):
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
    frozen_value: Any | None = None
    encoding: EncodingConfig = EncodingConfig()

    @property
    def encoding_mismatch(self) -> bool:
        return self.encoding.from_ != self.encoding.to

    @property
    def is_color_input(self) -> bool:
        return self.encoding.from_ in {Encoding.RGB, Encoding.BGR}

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
    def _validate_encoding(cls, data: dict[str, Any]) -> dict[str, Any]:
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
    def _random_calibration(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("calibration") in ["random", None]:
            data["calibration"] = RandomCalibrationConfig()
        return data

    @field_validator("scale_values", mode="before")
    @staticmethod
    def _parse_scale_values(value: Any) -> Any:
        """Parses the scale_values from the config."""
        return InputConfig._parse_values("scale", value)

    @field_validator("mean_values", mode="before")
    @staticmethod
    def _parse_mean_values(value: Any) -> Any:
        """Parses the mean_values from the config."""
        return InputConfig._parse_values("mean", value)

    @staticmethod
    def _parse_values(
        values_type: Literal["mean", "scale"], value: Any
    ) -> Any:
        """Resolves named values from the config."""
        if value is None:
            return None
        if isinstance(value, str) and value in NAMED_VALUES:
            return NAMED_VALUES[value][values_type]
        if isinstance(value, float | int):
            return [value, value, value]
        return value


class TargetConfig(CustomBaseModel):
    disable_calibration: bool = False


class HailoConfig(TargetConfig):
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] = 2
    compression_level: Literal[0, 1, 2, 3, 4, 5] = 2
    batch_size: int = 8
    disable_compilation: bool = False
    alls: list[str] = []
    hw_arch: Literal[
        "hailo8", "hailo8l", "hailo8r", "hailo10h", "hailo15h", "hailo15m"
    ] = "hailo8"


class BlobBaseConfig(TargetConfig):
    mo_args: list[str] = []
    compile_tool_args: list[str] = []
    compress_to_fp16: bool = True


class RVC2Config(BlobBaseConfig):
    number_of_shaves: int = 8
    superblob: bool = True

    @model_validator(mode="after")
    def _validate_superblob(self) -> Self:
        if self.superblob and self.number_of_shaves != 8:
            logger.warning("Changing number_of_shaves to 8 for superblob.")
            self.number_of_shaves = 8

        return self


class RVC3Config(BlobBaseConfig):
    pot_target_device: PotDevice = PotDevice.VPU


class RVC4Config(TargetConfig):
    compress_to_fp16: bool = False
    snpe_onnx_to_dlc_args: list[str] = []
    snpe_dlc_quant_args: list[str] = []
    snpe_dlc_graph_prepare_args: list[str] = []
    keep_raw_images: bool = False
    use_per_channel_quantization: bool = True
    use_per_row_quantization: bool = False
    optimization_level: Literal[1, 2, 3] = 2
    htp_socs: list[
        Literal["sm8350", "sm8450", "sm8550", "sm8650", "qcs6490", "qcs8550"]
    ] = ["sm8550"]

    @model_validator(mode="after")
    def _validate_fp16(self) -> Self:
        if not self.compress_to_fp16:
            return self
        self.disable_calibration = True
        if "qcs8550" not in self.htp_socs:
            self.htp_socs.append("qcs8550")
        return self


class SingleStageConfig(CustomBaseModel):
    input_model: Path
    input_bin: Path | None = None
    input_file_type: InputFileType

    inputs: Annotated[list[InputConfig], Field(min_length=1)] = []
    outputs: Annotated[list[OutputConfig], Field(min_length=1)] = []

    keep_intermediate_outputs: bool = True
    disable_onnx_simplification: bool = False
    disable_onnx_optimization: bool = False
    output_remote_url: str | None = None
    put_file_plugin: str | None = None

    hailo: HailoConfig = HailoConfig()
    rvc2: RVC2Config = RVC2Config()
    rvc3: RVC3Config = RVC3Config()
    rvc4: RVC4Config = RVC4Config()

    def get_target_config(self, target: Target) -> TargetConfig:
        """Returns the target configuration for the given target."""
        if target == Target.HAILO:
            return self.hailo
        if target == Target.RVC2:
            return self.rvc2
        if target == Target.RVC3:
            return self.rvc3
        if target == Target.RVC4:
            return self.rvc4

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, data: dict[str, Any]) -> dict[str, Any]:
        mean_values = data.pop("mean_values", None)
        scale_values = data.pop("scale_values", None)
        encoding = data.pop("encoding", {})
        data_type = data.pop("data_type", None)
        shape = data.pop("shape", None)
        layout = data.pop("layout", None)
        top_level_calibration = data.pop("calibration", {})

        input_file_type = InputFileType.from_path(data["input_model"])
        data["input_file_type"] = input_file_type
        if input_file_type == InputFileType.PYTORCH:
            logger.info(
                "Detected PyTorch model. Only YOLO models are supported."
            )
            input_shape = data.pop("yolo_input_shape", [640, 640])
            if isinstance(input_shape, str):
                input_shape = (
                    list(map(int, input_shape.split(" ")))
                    if " " in input_shape
                    else [int(input_shape)] * 2
                )
            else:
                logger.warning(
                    "yolo_input_shape is not provided. Using default shape [640, 640]."
                )
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
            metadata = get_metadata(Path(data["input_model"]))

        inputs = data.get("inputs")
        if not inputs:
            inputs = [
                {"name": cast(Any, name)} for name in metadata.input_shapes
            ]
        outputs = data.get("outputs")
        if not outputs:
            outputs = [
                {"name": cast(Any, name)} for name in metadata.output_shapes
            ]

        for inp in inputs:
            if "name" not in inp:
                raise ValueError(
                    f"Unable to determine name for input: `{inp}`."
                )
            inp_name = str(inp["name"])
            if inp_name not in metadata.input_shapes:
                tensor_shape, tensor_dtype = _get_onnx_inter_info(
                    data["input_model"], inp_name
                )
                metadata.input_shapes[inp_name] = tensor_shape  # type: ignore
                metadata.input_dtypes[inp_name] = tensor_dtype  # type: ignore
                logger.warning(
                    f"Input `{inp_name}` is not present in inputs of the ONNX model. "
                    f"Assuming it is an intermediate node."
                )
            onnx_shape, onnx_dtype = (
                metadata.input_shapes[inp_name],
                metadata.input_dtypes[inp_name],
            )
            inp["shape"] = inp.get("shape") or shape or onnx_shape
            inp["layout"] = inp.get("layout") or layout
            inp["data_type"] = inp.get("data_type") or data_type or onnx_dtype
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

            inp_calibration: dict[str, Any] = inp.get("calibration", {})
            if not inp_calibration and not top_level_calibration:
                inp["calibration"] = None
            elif top_level_calibration == "random":
                inp["calibration"] = "random"
            else:
                inp["calibration"] = {
                    **top_level_calibration,
                    **inp_calibration,
                }

        for out in outputs:
            out_name = str(out["name"])
            if (
                out_name not in metadata.output_shapes
                and out.get("data_type") is None
                and out.get("shape") is None
            ):
                tensor_shape, tensor_dtype = _get_onnx_inter_info(
                    data["input_model"], out_name
                )
                onnx_shape, onnx_dtype = tensor_shape, tensor_dtype
            elif out_name in metadata.output_shapes:
                onnx_shape, onnx_dtype = (
                    metadata.output_shapes[out_name],
                    metadata.output_dtypes[out_name],
                )
            else:
                onnx_shape, onnx_dtype = None, None
            out["shape"] = out.get("shape") or onnx_shape
            out["data_type"] = out.get("data_type") or onnx_dtype

        data["inputs"] = inputs
        data["outputs"] = outputs

        disable_calibration: bool | None = data.pop(
            "disable_calibration", None
        )
        if disable_calibration is None:
            return data

        for target in ["hailo", "rvc2", "rvc3", "rvc4"]:
            if target not in data:
                data[target] = {}

            data[target]["disable_calibration"] = disable_calibration
        return data

    @model_validator(mode="before")
    @classmethod
    def _download_input_model(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "input_model" not in value:
            raise ValueError("`input_model` must be provided.")
        input_file_type = InputFileType.from_path(value["input_model"])
        if input_file_type == InputFileType.IR:
            bin_path, xml_path = _extract_bin_xml_from_ir(
                value.get("input_model")
            )
            value["input_bin"] = bin_path
            value["input_model"] = xml_path
        else:
            value["input_model"] = resolve_path(
                value["input_model"], MODELS_DIR
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
    def _validate_name(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("name") is None:
            data["name"] = "-".join(data["stages"].keys())
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_stages(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "stages" not in data:
            name = data.pop("name", "default_stage")
            rich_logging = data.pop("rich_logging", True)
            data = {
                "name": name,
                "rich_logging": rich_logging,
                "stages": {name: data},
            }
        else:
            extra = {}
            for key in list(data.keys()):
                if key not in cls.model_fields:
                    extra[key] = data.pop(key)
            for stage in data["stages"].values():
                for key, value in extra.items():
                    if key not in stage:
                        stage[key] = value
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


def _extract_bin_xml_from_ir(ir_path: Any) -> tuple[Path, Path]:
    """Extracts the corresponding second path from a single IR path.

    We assume that the base filename matches between the .bin and .xml
    file. Otherwise, an error will be thrown.
    """

    if not isinstance(ir_path, str) and not isinstance(ir_path, Path):
        raise TypeError("`input_path` must be str or Path.")
    ir_path = Path(ir_path)

    if ir_path.suffix == ".bin":
        bin_path = ir_path
        xml_path = str(bin_path.with_suffix(".xml"))
    elif ir_path.suffix == ".xml":
        xml_path = ir_path
        bin_path = str(xml_path.with_suffix(".bin"))
    else:
        raise ValueError(
            "`ir_path` is invalid: does not have .bin or .xml extension."
        )

    # fix any remote path corruption from pathlib
    bin_path = str(bin_path).replace(":/", "://")
    xml_path = str(xml_path).replace(":/", "://")

    try:
        bin_path = resolve_path(bin_path, MODELS_DIR)
    except Exception as e:
        raise ValueError(
            f"`bin_path` {bin_path} was not found. "
            "Please ensure that your xml and bin file have matching file basenames "
            "and are located in the same directory."
        ) from e
    try:
        xml_path = resolve_path(xml_path, MODELS_DIR)
    except Exception as e:
        raise ValueError(
            f"`xml_path` {xml_path} was not found. "
            "Please ensure that your xml and bin file have matching file basenames and "
            "are located in the same directory."
        ) from e

    return bin_path, xml_path


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

    shape = [
        dim.dim_value for dim in output_value_info.type.tensor_type.shape.dim
    ]
    if any(dim == 0 for dim in shape):
        raise ValueError(
            "Dynamic shapes are not supported. "
            f"Shape of node '{node_name}' is {shape}."
        )
    data_type = output_value_info.type.tensor_type.elem_type

    return shape, DataType.from_onnx_dtype(data_type)


def _get_onnx_tensor_info(
    model_path: Path | str, tensor_name: str
) -> tuple[list[int], DataType]:
    model = onnx.load(str(model_path))

    def extract_tensor_info(
        tensor_type: TypeProto.Tensor,
    ) -> tuple[list[int], DataType]:
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        if any(dim == 0 for dim in shape):
            raise ValueError(
                "Dynamic shapes are not supported. "
                f"Shape of tensor '{tensor_name}' is {shape}."
            )
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
    if onnx_path.with_suffix(".onnx_data").exists():
        model_data_path = onnx_path.with_suffix(".onnx_data")
    else:
        model_data_path = None

    for node in model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in rename_dict:
                node.input[i] = rename_dict[input_name]

        for i, output_name in enumerate(node.output):
            if output_name in rename_dict:
                node.output[i] = rename_dict[output_name]

    if model_data_path:
        onnx.save(
            model,
            str(output_path),
            save_as_external_data=True,
            location=f"{output_path.name}_data",
        )
    else:
        onnx.save(model, str(output_path))
