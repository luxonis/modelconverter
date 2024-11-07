from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
from onnx.onnx_pb import TensorProto

__all__ = ["Encoding", "DataType", "ResizeMethod", "PotDevice", "Target"]


class Layout(Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"


class Encoding(Enum):
    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"
    NONE = "NONE"


class DataType(Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BOOLEAN = "boolean"
    STRING = "string"
    UFXP8 = "ufxp8"
    UFXP16 = "ufxp16"
    UFXP32 = "ufxp32"
    UFXP64 = "ufxp64"
    FXP8 = "fxp8"
    FXP16 = "fxp16"
    FXP32 = "fxp32"
    FXP64 = "fxp64"

    @classmethod
    def from_tensorflow_dtype(cls, dtype: int) -> "DataType":
        from tflite.TensorType import TensorType

        tensor_types = {
            TensorType.FLOAT16: "float16",
            TensorType.FLOAT32: "float32",
            TensorType.FLOAT64: "float64",
            TensorType.INT16: "int16",
            TensorType.INT32: "int32",
            TensorType.INT64: "int64",
            TensorType.UINT8: "uint8",
            TensorType.UINT16: "uint16",
            TensorType.UINT32: "uint32",
            TensorType.BOOL: "boolean",
            TensorType.STRING: "string",
        }

        if dtype not in tensor_types:
            raise ValueError(f"Unsupported TensorFlow data type: `{dtype}`")
        return cls(tensor_types[dtype])

    @classmethod
    def from_dlc_dtype(cls, dtype: str) -> "DataType":
        dtype_map = {
            "Float_16": "float16",
            "Float_32": "float32",
            "Float_64": "float64",
            "Int_8": "int8",
            "Int_16": "int16",
            "Int_32": "int32",
            "Int_64": "int64",
            "uInt_8": "uint8",
            "uInt_16": "uint16",
            "uInt_32": "uint32",
            "uInt_64": "uint64",
            "uFxp_8": "ufxp8",
            "uFxp_16": "ufxp16",
            "uFxp_32": "ufxp32",
            "Fxp_8": "fxp8",
            "Fxp_16": "fxp16",
            "Fxp_32": "fxp32",
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported DLC data type: `{dtype}`")
        return cls(dtype_map[dtype])

    @classmethod
    def from_onnx_dtype(cls, dtype: int) -> "DataType":
        dtype_map = {
            TensorProto.FLOAT16: "float16",
            TensorProto.FLOAT: "float32",
            TensorProto.DOUBLE: "float64",
            TensorProto.UINT8: "uint8",
            TensorProto.UINT16: "uint16",
            TensorProto.UINT32: "uint32",
            TensorProto.UINT64: "uint64",
            TensorProto.INT8: "int8",
            TensorProto.INT16: "int16",
            TensorProto.INT32: "int32",
            TensorProto.INT64: "int64",
            TensorProto.BOOL: "boolean",
            TensorProto.STRING: "string",
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported ONNX data type: `{dtype}`")
        return cls(dtype_map[dtype])

    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> "DataType":
        dtype_map = {
            np.float16: "float16",
            np.float32: "float32",
            np.float64: "float64",
            np.int8: "int8",
            np.int16: "int16",
            np.int32: "int32",
            np.int64: "int64",
            np.uint8: "uint8",
            np.uint16: "uint16",
            np.uint32: "uint32",
            np.uint64: "uint64",
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported numpy data type: `{dtype}`")
        return cls(dtype_map[dtype])

    @classmethod
    def from_ir_ie_dtype(cls, dtype: str) -> "DataType":
        dtype_map = {
            "FP16": "float16",
            "FP32": "float32",
            "FP64": "float64",
            "I8": "int8",
            "I16": "int16",
            "I32": "int32",
            "I64": "int64",
            "U8": "uint8",
            "U16": "uint16",
            "U32": "uint32",
            "U64": "uint64",
            "BOOL": "boolean",
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported IR data type: `{dtype}`")
        return cls(dtype_map[dtype])

    @classmethod
    def from_ir_runtime_dtype(cls, dtype: str) -> "DataType":
        dtype_map = {
            "f16": "float16",
            "f32": "float32",
            "f64": "float64",
            "u8": "uint8",
            "u16": "uint16",
            "u32": "uint32",
            "u64": "uint64",
            "i8": "int8",
            "i16": "int16",
            "i32": "int32",
            "i64": "int64",
            "boolean": "boolean",
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported IR runtime data type: `{dtype}`")
        return cls(dtype_map[dtype])

    def as_numpy_dtype(self) -> np.dtype:
        return {
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
        }[self.value]

    def as_openvino_dtype(self) -> str:
        return {
            "float16": "f16",
            "float32": "f32",
            "float64": "f64",
            "int8": "i8",
            "int16": "i16",
            "int32": "i32",
            "int64": "i64",
            "uint8": "u8",
            "uint16": "u16",
            "uint32": "u32",
            "uint64": "u64",
        }[self.value]

    def as_snpe_dtype(self) -> str:
        return self.value


class ResizeMethod(Enum):
    CROP = "CROP"
    PAD = "PAD"
    RESIZE = "RESIZE"


class PotDevice(Enum):
    VPU = "VPU"
    ANY = "ANY"


class Target(Enum):
    HAILO = "hailo"
    RVC2 = "rvc2"
    RVC3 = "rvc3"
    RVC4 = "rvc4"


class InputFileType(Enum):
    ONNX = "ONNX"
    IR = "IR"
    TFLITE = "TFLITE"
    DLC = "DLC"
    HAR = "HAR"

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "InputFileType":
        path = Path(path)
        if path.suffix == ".onnx":
            return cls.ONNX
        if path.suffix in [".xml", ".bin"]:
            return cls.IR
        if path.suffix == ".tflite":
            return cls.TFLITE
        if path.suffix == ".dlc":
            return cls.DLC
        if path.suffix == ".har":
            return cls.HAR
        raise ValueError(f"Unsupported file type: `{path}`")
