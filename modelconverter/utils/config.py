import logging
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import onnx
from luxonis_ml.utils import LuxonisConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated, Self, TypeAlias

from modelconverter.utils.calibration_data import download_calibration_data
from modelconverter.utils.constants import MODELS_DIR
from modelconverter.utils.filesystem_utils import resolve_path
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    PotDevice,
    ResizeMethod,
)

logger = logging.getLogger(__name__)

FileInfoType: TypeAlias = Dict[
    str, Tuple[Optional[List[Optional[int]]], Optional[DataType]]
]

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
    output: Optional[str] = None
    script: Optional[str] = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.output is None and self.script is None:
            raise ValueError(
                "Either `output` or `script` must be provided for calibration."
            )
        return self

    @field_validator("script", mode="after")
    @staticmethod
    def _download_calibration_script(script: Any) -> Optional[Path]:
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
    def _download_calibration_data(value: Any) -> Optional[Path]:
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
    shape: Optional[List[Optional[int]]] = None
    data_type: DataType = DataType.FLOAT32

    @field_validator("data_type", mode="before")
    @staticmethod
    def _default_data_type(value: Any) -> DataType:
        """Parses the data_type from the config."""
        if value is None:
            return DataType.FLOAT32
        return DataType(value)


class EncodingConfig(CustomBaseModel):
    from_: Annotated[
        Encoding, Field(alias="from", serialization_alias="from")
    ] = Encoding.RGB
    to: Encoding = Encoding.BGR


class InputConfig(OutputConfig):
    calibration: Union[
        ImageCalibrationConfig, RandomCalibrationConfig, LinkCalibrationConfig
    ] = RandomCalibrationConfig()
    scale_values: Optional[Annotated[List[float], Field(min_length=1)]] = None
    mean_values: Optional[Annotated[List[float], Field(min_length=1)]] = None
    reverse_input_channels: bool = False
    frozen_value: Optional[Any] = None
    encoding: EncodingConfig = EncodingConfig()

    @model_validator(mode="before")
    @classmethod
    def _validate_encoding(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        encoding = data.get("encoding")
        if encoding is None:
            return data
        if isinstance(encoding, str):
            data["encoding"] = {"from": encoding, "to": encoding}
        return data

    @model_validator(mode="after")
    def _validate_reverse_input_channels(self) -> Self:
        if self.reverse_input_channels:
            return self

        if self.encoding.from_ == Encoding.NONE:
            self.reverse_input_channels = False
            return self

        if (
            self.encoding.from_ == Encoding.GRAY
            or self.encoding.to == Encoding.GRAY
        ):
            self.encoding.from_ = self.encoding.to = Encoding.GRAY
            self.reverse_input_channels = False

        if self.encoding.from_ != self.encoding.to:
            self.reverse_input_channels = True

        return self

    @model_validator(mode="before")
    @classmethod
    def _random_calibration(cls, data: Dict[str, Any]) -> Dict[str, Any]:
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
        if isinstance(value, str):
            if value in NAMED_VALUES:
                return NAMED_VALUES[value][values_type]
        if isinstance(value, (float, int)):
            return [value, value, value]
        return value

    @model_validator(mode="before")
    @classmethod
    def _validate_deprecated_reverse_channels(
        cls, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if "reverse_input_channels" not in data:
            return data
        logger.warning(
            "Field `reverse_input_channels` is deprecated. "
            "Please use `encoding.from` and `encoding.to` instead."
        )
        reverse = data["reverse_input_channels"]
        calib = data.get("calibration", {}) or {}
        if isinstance(calib, str):
            calib_encoding = Encoding.BGR
        else:
            calib_encoding = calib.get("encoding", Encoding.BGR)

        if reverse:
            if calib_encoding == Encoding.GRAY:
                raise ValueError(
                    "Cannot reverse channels for grayscale images."
                )
            else:
                encoding = {"from": Encoding.RGB, "to": Encoding.BGR}
        else:
            if calib_encoding == Encoding.GRAY:
                encoding = "GRAY"
            else:
                encoding = {"from": Encoding.BGR, "to": Encoding.BGR}

        data["encoding"] = encoding
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_deprecated_reverse_calib_encoding(
        cls, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        calib = data.get("calibration", {}) or {}
        if isinstance(calib, str):
            return data

        calib_encoding = calib.pop("encoding", None)
        if calib_encoding is None:
            return data

        logger.warning(
            "Field `calibration.encoding` is deprecated. Please use `encoding.to` instead."
        )
        encoding = data.get("encoding", {})
        if isinstance(encoding, str):
            encoding = {"from": encoding, "to": encoding}
        encoding["to"] = calib_encoding
        data["encoding"] = encoding
        return data


class TargetConfig(CustomBaseModel):
    disable_calibration: bool = False


class HailoConfig(TargetConfig):
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] = 2
    compression_level: Literal[0, 1, 2, 3, 4, 5] = 2
    batch_size: int = 8
    early_stop: bool = False
    alls: List[str] = []


class BlobBaseConfig(TargetConfig):
    mo_args: List[str] = []
    compile_tool_args: List[str] = []


class RVC2Config(BlobBaseConfig):
    number_of_shaves: int = 8
    number_of_cmx_slices: int = 8
    superblob: bool = True

    @model_validator(mode="after")
    def _validate_cmx_slices(self) -> Self:
        if self.superblob:
            logger.info("Superblob enabled. Setting number of shaves to 8.")
            self.number_of_cmx_slices = self.number_of_shaves = 8

        elif self.number_of_cmx_slices < self.number_of_shaves:
            logger.warning(
                "Number of CMX slices must be greater than or equal "
                "to the number of shaves. "
                "Setting `number_of_cmx_slices` to "
                f"`number_of_shaves` ({self.number_of_shaves})."
            )
            self.number_of_cmx_slices = self.number_of_shaves
        return self


class RVC3Config(BlobBaseConfig):
    pot_target_device: PotDevice = PotDevice.VPU


class RVC4Config(TargetConfig):
    snpe_onnx_to_dlc_args: List[str] = []
    snpe_dlc_quant_args: List[str] = []
    snpe_dlc_graph_prepare_args: List[str] = []
    keep_raw_images: bool = False
    htp_socs: List[
        Literal["sm8350", "sm8450", "sm8550", "sm8650", "qcs6490", "qcs8550"]
    ] = ["sm8550"]


class SingleStageConfig(CustomBaseModel):
    input_model: Path
    input_bin: Optional[Path] = None
    input_file_type: InputFileType

    inputs: Annotated[List[InputConfig], Field(min_length=1)] = []
    outputs: Annotated[List[OutputConfig], Field(min_length=1)] = []

    keep_intermediate_outputs: bool = True
    disable_onnx_simplification: bool = False
    output_remote_url: Optional[str] = None
    put_file_plugin: Optional[str] = None

    hailo: HailoConfig = HailoConfig()
    rvc2: RVC2Config = RVC2Config()
    rvc3: RVC3Config = RVC3Config()
    rvc4: RVC4Config = RVC4Config()

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        mean_values = data.pop("mean_values", None)
        scale_values = data.pop("scale_values", None)
        encoding = data.pop("encoding", {})
        data_type = data.pop("data_type", None)
        shape = data.pop("shape", None)
        reverse_input_channels = data.pop("reverse_input_channels", None)
        top_level_calibration = data.pop("calibration", {})

        input_file_type = _detect_input_file_type(data["input_model"])
        data["input_file_type"] = input_file_type
        file_inputs: FileInfoType = {}
        file_outputs: FileInfoType = {}
        if input_file_type == InputFileType.ONNX:
            file_inputs, file_outputs = _get_onnx_info(data["input_model"])
        if input_file_type == InputFileType.TFLITE:
            file_inputs, file_outputs = _get_tflite_info(data["input_model"])
        elif input_file_type == InputFileType.IR:
            file_inputs, file_outputs = _get_ir_info(
                data["input_bin"], data["input_model"]
            )

        inputs = data.get("inputs")
        if not inputs:
            inputs = [{"name": cast(Any, name)} for name in file_inputs.keys()]
        outputs = data.get("outputs")
        if not outputs:
            outputs = [
                {"name": cast(Any, name)} for name in file_outputs.keys()
            ]

        for inp in inputs:
            if "name" not in inp:
                raise ValueError(
                    f"Unable to determine name for input: `{inp}`."
                )
            inp_name = str(inp["name"])
            if inp_name not in file_inputs:
                tensor_shape, tensor_dtype = _get_onnx_inter_info(
                    data["input_model"], inp_name
                )
                file_inputs[inp_name] = tensor_shape, tensor_dtype
                logger.warning(
                    f"Input `{inp_name}` is not present in inputs of the ONNX model. "
                    f"Assuming it is an intermediate node."
                )
            onnx_shape, onnx_dtype = file_inputs[inp_name]
            inp["shape"] = inp.get("shape") or shape or onnx_shape
            inp["data_type"] = inp.get("data_type") or data_type or onnx_dtype
            inp["encoding"] = inp.get("encoding") or encoding
            inp["mean_values"] = inp.get("mean_values") or mean_values
            inp["scale_values"] = inp.get("scale_values") or scale_values

            if (
                inp.get("reverse_input_channels") is not None
                or reverse_input_channels
            ):
                inp["reverse_input_channels"] = inp.get(
                    "reverse_input_channels"
                ) or (reverse_input_channels or False)

            inp_calibration: Dict[str, Any] = inp.get("calibration", {})
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
                out_name not in file_outputs
                and out.get("data_type") is None
                and out.get("shape") is None
            ):
                tensor_shape, tensor_dtype = _get_onnx_inter_info(
                    data["input_model"], out_name
                )
                onnx_shape, onnx_dtype = tensor_shape, tensor_dtype
            elif out_name in file_outputs:
                onnx_shape, onnx_dtype = file_outputs[out_name]
            else:
                onnx_shape, onnx_dtype = None, None
            out["shape"] = out.get("shape") or onnx_shape
            out["data_type"] = out.get("data_type") or onnx_dtype

        data["inputs"] = inputs
        data["outputs"] = outputs

        disable_calibration: Optional[bool] = data.pop(
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
    def _download_input_model(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if "input_model" not in value:
            raise ValueError("`input_model` must be provided.")
        input_file_type = _detect_input_file_type(value["input_model"])
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
    stages: Annotated[Dict[str, SingleStageConfig], Field(min_length=1)]
    output_dir_name: Optional[str] = None
    name: str

    def get_stage_config(self, stage: Optional[str]) -> SingleStageConfig:
        if stage is None:
            if len(self.stages) == 1:
                return next(iter(self.stages.values()))
            raise ValueError("Multiple stages found. Please specify a stage.")
        return self.stages[stage]

    @model_validator(mode="before")
    @classmethod
    def _validate_name(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get("name") is None:
            data["name"] = "-".join(data["stages"].keys())
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_stages(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if "stages" not in data:
            name = data.pop("name", "default_stage")
            output_dir_name = data.pop("output_dir_name", None)
            data = {
                "stages": {name: data},
                "name": name,
                "output_dir_name": output_dir_name,
            }
        else:
            extra = {}
            for key in list(data.keys()):
                if key not in cls.__fields__:
                    extra[key] = data.pop(key)

            for stage in data["stages"].values():
                for key, value in extra.items():
                    if key not in stage:
                        stage[key] = value

        return data

    @model_validator(mode="after")
    def _validate_single_stage_name(self) -> Self:
        """Changes the default 'default_stage' name to the name of the input model."""
        if len(self.stages) == 1:
            stage = next(iter(self.stages.values()))
            model_name = stage.input_model.stem
            self.stages = {model_name: stage}
            self.name = model_name
        return self


def _detect_input_file_type(input_path: Union[str, Path]) -> InputFileType:
    if not isinstance(input_path, str) and not isinstance(input_path, Path):
        raise ValueError("`input_path` must be str or Path.")
    input_path = Path(input_path)

    if input_path.suffix == ".onnx":
        return InputFileType.ONNX
    elif input_path.suffix == ".tflite":
        return InputFileType.TFLITE
    elif input_path.suffix == ".pt":
        raise NotImplementedError("PyTorch (.pt) is not yet supported.")
    elif input_path.suffix in [".bin", ".xml"]:
        return InputFileType.IR
    else:
        raise ValueError("Input file format is not recognized.")


def _extract_bin_xml_from_ir(ir_path: Any) -> Tuple[Path, Path]:
    """Extracts the corresponding second path from a single IR path.

    We assume that the base filename matches between the .bin and .xml file. Otherwise,
    an error will be thrown.
    """

    if not isinstance(ir_path, str) and not isinstance(ir_path, Path):
        raise ValueError("`input_path` must be str or Path.")
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


def _get_tflite_info(
    tflite_path: Path,
) -> Tuple[FileInfoType, FileInfoType]:
    """Reads names, shapes, and data types for all inputs and outputs of the provided
    TFLite model.

    Args:
        tflite_path (Path): Path to the TFLite model.

    Returns:
        Tuple[FileInfoType, FileInfoType]: (input_info, output_info) where the keys
        are the input names and the values are tuples of (shape, DataType).
    """

    import tflite

    with open(tflite_path, "rb") as f:
        data = f.read()

    # Load the model
    model = tflite.Model.GetRootAsModel(data, 0)

    # Get the subgraph (model usually contains only one subgraph)
    subgraph = model.Subgraphs(0)
    if subgraph is None:
        raise ValueError("Failed to load TFLite model.")

    input_info = {}
    output_info = {}

    for i in range(subgraph.InputsLength()):
        tensor = subgraph.Tensors(subgraph.Inputs(i))
        input_info[tensor.Name().decode("utf-8")] = (  # type: ignore
            tensor.ShapeAsNumpy().tolist(),  # type: ignore
            DataType.from_tensorflow_dtype(tensor.Type()),  # type: ignore
        )

    for i in range(subgraph.OutputsLength()):
        tensor = subgraph.Tensors(subgraph.Outputs(i))
        output_info[tensor.Name().decode("utf-8")] = (  # type: ignore
            tensor.ShapeAsNumpy().tolist(),  # type: ignore
            DataType.from_tensorflow_dtype(tensor.Type()),  # type: ignore
        )

    return input_info, output_info


def _get_onnx_info(onnx_path: Path) -> Tuple[FileInfoType, FileInfoType]:
    """Reads names, shapes and data types for all inputs and outputs of the provided
    ONNX model.

    Args:
        onnx_path (Path): Path to the ONNX model.

    Returns:
        Tuple[FileInfoType, FileInfoType]: (input_info, output_info) where the keys
        are the input names and the values are tuples of (shape, DataType).
    """

    try:
        model = onnx.load(str(onnx_path))
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model: `{onnx_path}`") from e

    input_info = {}
    output_info = {}

    for inp in model.graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        dtype = DataType.from_onnx_dtype(inp.type.tensor_type.elem_type)
        input_info[inp.name] = (shape, dtype)

    for output in model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        dtype = DataType.from_onnx_dtype(output.type.tensor_type.elem_type)
        output_info[output.name] = (shape, dtype)

    return input_info, output_info


def _get_ir_info(
    bin_path: Path, xml_path: Path
) -> Tuple[FileInfoType, FileInfoType]:
    """Reads names, shapes and data types for all inputs and outputs of the provided IR
    model (bin and xml).

    Args:
        bin_path (Path): Path to the OpenVINO binary weights of the model.
        xml_path (Path): Path to the OpenVINO XML definition of the model.

    Returns:
        Tuple[FileInfoType, FileInfoType]: (input_info, output_info) where the keys
        are the input names and the values are tuples of (shape, DataType).
    """

    from openvino.runtime import Core

    ie = Core()
    try:
        model = ie.read_model(model=str(xml_path), weights=str(bin_path))
    except Exception as e:
        raise ValueError(
            f"Failed to load IR model: `{bin_path}` and `{xml_path}`"
        ) from e

    input_info = {}
    output_info = {}

    for inp in model.inputs:
        name = list(inp.names)[0]
        dtype = DataType.from_ir_dtype(inp.element_type.get_type_name())
        input_info[name] = (inp.shape, dtype)
    for output in model.outputs:
        name = list(output.names)[0]
        dtype = DataType.from_ir_dtype(output.element_type.get_type_name())
        output_info[name] = (output.shape, dtype)

    return input_info, output_info


def _get_onnx_node_info(
    model_path: Path, node_name: str
) -> Tuple[List[Optional[int]], DataType]:
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
        dim.dim_value if dim.dim_value > 0 else None
        for dim in output_value_info.type.tensor_type.shape.dim
    ]
    data_type = output_value_info.type.tensor_type.elem_type

    return shape, DataType.from_onnx_dtype(data_type)


def _get_onnx_tensor_info(
    model_path: Union[Path, str], tensor_name: str
) -> Tuple[List[Optional[int]], DataType]:
    model = onnx.load(str(model_path))

    def extract_tensor_info(tensor_type):
        shape = [
            dim.dim_value if dim.dim_value > 0 else None
            for dim in tensor_type.shape.dim
        ]
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
) -> Tuple[Optional[List[Optional[int]]], Optional[DataType]]:
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
    onnx_path: Union[Path, str],
    rename_dict: Dict[str, str],
    output_path: Union[Path, str],
) -> None:
    model = onnx.load(str(onnx_path))

    for node in model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in rename_dict:
                node.input[i] = rename_dict[input_name]

        for i, output_name in enumerate(node.output):
            if output_name in rename_dict:
                node.output[i] = rename_dict[output_name]

    onnx.save(model, str(output_path))
