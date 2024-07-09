from enum import Enum

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
    def from_ir_dtype(cls, dtype: str) -> "DataType":
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
            raise ValueError(f"Unsupported IR data type: `{dtype}`")
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

    @property
    def suffix(self) -> str:
        return {
            "hailo": ".hef",
            "rvc2": ".blob",
            "rvc3": ".blob",
            "rvc4": ".dlc",
        }[self.value]


class InputFileType(Enum):
    ONNX = "ONNX"
    IR = "IR"
    TFLITE = "TFLITE"
