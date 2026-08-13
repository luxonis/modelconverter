"""Enumerations describing models and conversion targets.

Defines the platforms a model can be converted for and the data types,
color encodings, and resize methods used to describe model inputs and
outputs. `DataType` also carries the conversions between the internal
names used here and the data type names of the frameworks involved in a
conversion, such as ONNX, ``numpy``, DepthAI, OpenVINO, TensorFlow
Lite, and SNPE.
"""

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from onnx.onnx_pb import TensorProto

__all__ = ["DataType", "Encoding", "PotDevice", "ResizeMethod", "Target"]

if TYPE_CHECKING:
    import depthai as dai


class Layout(Enum):
    """Memory layout of a four-dimensional model input or output.

    ``NCHW`` is planar and ``NHWC`` interleaved.
    """

    NCHW = "NCHW"
    NHWC = "NHWC"


class Encoding(Enum):
    """Color encoding of the images fed to a model input.

    ``NONE`` marks an input that is not fed with image data and so has
    no color encoding at all.
    """

    RGB = "RGB"
    BGR = "BGR"
    GRAY = "GRAY"
    NONE = "NONE"


class DataType(Enum):
    """Data type of a model input or output.

    Covers the floating point, integer, boolean, string, and
    fixed-point types used by the supported frameworks, and provides
    the conversions from and to their own names for these types.
    """

    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT4 = "uint4"
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
        """Create a `DataType` from a TensorFlow Lite tensor type.

        Args:
            dtype: Member of the ``tflite.TensorType`` enum.

        Returns:
            The equivalent `DataType`.

        Raises:
            ValueError: If the type has no `DataType` equivalent.

        """
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
    def from_dai_dtype(cls, dtype: "dai.TensorInfo.DataType") -> "DataType":
        """Create a `DataType` from a DepthAI tensor data type.

        Args:
            dtype: Member of the ``dai.TensorInfo.DataType`` enum.

        Returns:
            The equivalent `DataType`.

        Raises:
            ValueError: If the type has no `DataType` equivalent.

        """
        import depthai as dai

        dtype_map = {
            dai.TensorInfo.DataType.FP16: "float16",
            dai.TensorInfo.DataType.FP32: "float32",
            dai.TensorInfo.DataType.FP64: "float64",
            dai.TensorInfo.DataType.I8: "int8",
            dai.TensorInfo.DataType.INT: "int32",
            dai.TensorInfo.DataType.U8F: "ufxp8",
            dai.TensorInfo.DataType.U16F: "ufxp16",
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported DepthAI data type: `{dtype}`")
        return cls(dtype_map[dtype])

    @classmethod
    def from_hubai_dtype(cls, dtype: str) -> "DataType":
        """Create a `DataType` from a HubAI data type name.

        Args:
            dtype: HubAI data type name, such as ``"FP16"``.

        Returns:
            The equivalent `DataType`.

        Raises:
            ValueError: If the name has no `DataType` equivalent.

        """
        dtype_map = {
            "INT8": "int8",
            "INT32": "int32",
            "FP16": "float16",
            "FP32": "float32",
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported HubAI data type: `{dtype}`")
        return cls(dtype_map[dtype])

    @classmethod
    def from_dlc_dtype(cls, dtype: str) -> "DataType":
        """Create a `DataType` from a DLC data type name.

        Args:
            dtype: DLC data type name, such as ``"Float_32"``.

        Returns:
            The equivalent `DataType`.

        Raises:
            ValueError: If the name has no `DataType` equivalent.

        """
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
        """Create a `DataType` from an ONNX tensor element type.

        Args:
            dtype: Member of the ``onnx.TensorProto`` data type enum.

        Returns:
            The equivalent `DataType`.

        Raises:
            ValueError: If the type has no `DataType` equivalent.

        """
        dtype_map = {
            TensorProto.BFLOAT16: "bfloat16",
            TensorProto.FLOAT16: "float16",
            TensorProto.FLOAT: "float32",
            TensorProto.DOUBLE: "float64",
            TensorProto.INT4: "int4",
            TensorProto.UINT8: "uint8",
            TensorProto.UINT4: "uint4",
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
        """Create a `DataType` from a ``numpy`` data type.

        Args:
            dtype: ``numpy`` scalar type, such as ``numpy.float32``.

        Returns:
            The equivalent `DataType`.

        Raises:
            ValueError: If the type has no `DataType` equivalent.

        """
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
        """Create a `DataType` from an OpenVINO IR data type name.

        Args:
            dtype: OpenVINO Inference Engine precision name, such as
                ``"FP16"``.

        Returns:
            The equivalent `DataType`.

        Raises:
            ValueError: If the name has no `DataType` equivalent.

        """
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
        """Create a `DataType` from an OpenVINO runtime type name.

        Args:
            dtype: Name of an OpenVINO runtime element type, such as
                ``"f32"``.

        Returns:
            The equivalent `DataType`.

        Raises:
            ValueError: If the name has no `DataType` equivalent.

        """
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

    def as_dai_dtype(self) -> "dai.TensorInfo.DataType":
        """Convert to the equivalent DepthAI data type.

        Returns:
            The matching member of ``dai.TensorInfo.DataType``.

        Raises:
            ValueError: If the data type has no DepthAI equivalent.

        """
        import depthai as dai

        return self._transform(
            {
                "float16": dai.TensorInfo.DataType.FP16,
                "float32": dai.TensorInfo.DataType.FP32,
                "float64": dai.TensorInfo.DataType.FP64,
                "int8": dai.TensorInfo.DataType.I8,
                "int32": dai.TensorInfo.DataType.INT,
                "uint8": dai.TensorInfo.DataType.U8F,
                "uint16": dai.TensorInfo.DataType.U16F,
                "ufxp8": dai.TensorInfo.DataType.U8F,
                "ufxp16": dai.TensorInfo.DataType.U16F,
            },
            "DepthAI",
        )

    def supports_dai_dtype(self) -> bool:
        """Check whether the data type has a DepthAI equivalent.

        Returns:
            ``True`` if `as_dai_dtype` can convert this data type,
            ``False`` otherwise.

        Example:
            >>> DataType.FLOAT32.supports_dai_dtype()
            True
            >>> DataType.STRING.supports_dai_dtype()
            False

        """
        return self.value in {
            "float16",
            "float32",
            "float64",
            "int8",
            "int32",
            "uint8",
            "uint16",
            "ufxp8",
            "ufxp16",
        }

    def as_numpy_dtype(self) -> np.dtype:
        """Convert to the closest ``numpy`` data type.

        Types without an exact counterpart are widened: ``bfloat16``
        becomes ``float32``, the 4-bit types become their 8-bit
        counterparts, and the fixed-point types become integers of the
        same width and signedness.

        Returns:
            The matching ``numpy`` scalar type.

        Example:
            >>> DataType.FLOAT32.as_numpy_dtype()
            <class 'numpy.float32'>
            >>> DataType.BFLOAT16.as_numpy_dtype()
            <class 'numpy.float32'>
            >>> DataType.INT4.as_numpy_dtype()
            <class 'numpy.int8'>

        """
        return self._transform(
            {
                "bfloat16": np.float32,  # Preserve bfloat16 range better than float16.
                "float16": np.float16,
                "float32": np.float32,
                "float64": np.float64,
                "int4": np.int8,  # NumPy has no 4-bit signed integer dtype.
                "int8": np.int8,
                "int16": np.int16,
                "int32": np.int32,
                "int64": np.int64,
                "uint4": np.uint8,  # NumPy has no 4-bit unsigned integer dtype.
                "uint8": np.uint8,
                "uint16": np.uint16,
                "uint32": np.uint32,
                "uint64": np.uint64,
                "boolean": np.bool_,
                "string": np.str_,
                "ufxp8": np.uint8,  # No fixed-point dtype in NumPy, so use uint8 as a placeholder.
                "ufxp16": np.uint16,
                "ufxp32": np.uint32,
                "ufxp64": np.uint64,
                "fxp8": np.int8,
                "fxp16": np.int16,
                "fxp32": np.int32,
                "fxp64": np.int64,
            },
            "numpy",
        )

    def as_openvino_dtype(self) -> str:
        """Convert to the equivalent OpenVINO data type name.

        Returns:
            The OpenVINO runtime element type name, such as ``"f32"``.

        Raises:
            ValueError: If the data type has no OpenVINO equivalent.

        """
        return self._transform(
            {
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
            },
            "OpenVINO",
        )

    def as_snpe_dtype(self) -> str:
        """Convert to the equivalent SNPE data type name.

        Returns:
            The value of the enum member, which SNPE uses unchanged.

        """
        return self.value

    def as_nn_archive_dtype(self) -> str:
        """Convert to the equivalent NN Archive data type name.

        Fixed-point types are reported as integers of the same width
        and signedness; every other type keeps its own name.

        Returns:
            The data type name to store in an NN Archive.

        Example:
            >>> DataType.UFXP8.as_nn_archive_dtype()
            'uint8'
            >>> DataType.FXP8.as_nn_archive_dtype()
            'int8'
            >>> DataType.FLOAT32.as_nn_archive_dtype()
            'float32'

        """
        if self.value.startswith("ufxp"):
            return self.value.replace("ufxp", "uint")
        if self.value.startswith("fxp"):
            return self.value.replace("fxp", "int")
        return self.value

    def _transform(self, mapping: dict[str, Any], desc: str) -> Any:
        if self.value not in mapping:
            raise ValueError(
                f"`{self.value}` cannot be transformed to {desc} data type"
            )
        return mapping[self.value]


class ResizeMethod(Enum):
    """Way of fitting a calibration image to the model input size.

    ``CROP`` cuts out the center of the image, ``RESIZE`` stretches it
    to the requested size, and ``PAD`` scales it while keeping its
    aspect ratio and pads the rest with black.
    """

    CROP = "CROP"
    PAD = "PAD"
    RESIZE = "RESIZE"


class PotDevice(Enum):
    """Target device of the OpenVINO POT quantization.

    Selects the device the Post-training Optimization Tool quantizes
    for when converting for RVC3.
    """

    VPU = "VPU"
    ANY = "ANY"


class Target(Enum):
    """Platform a model is converted for.

    Each target has its own conversion package and Docker image.
    """

    HAILO = "hailo"
    RVC2 = "rvc2"
    RVC3 = "rvc3"
    RVC4 = "rvc4"


class QuantizationMode(Enum):
    """Precision a model is quantized to.

    The accuracy-focused and mixed variants trade throughput for
    accuracy. ``CUSTOM`` leaves the quantization to the arguments given
    in the configuration instead of picking a preset.
    """

    INT8_STD = "INT8_STANDARD"
    INT8_ACC = "INT8_ACCURACY_FOCUSED"
    INT8_16_MIX = "INT8_INT16_MIXED"
    INT8_16_MIX_ACC = "INT8_INT16_MIXED_ACCURACY_FOCUSED"
    FP16_STD = "FP16_STANDARD"
    CUSTOM = "CUSTOM"


class InputFileType(Enum):
    """Format of the model given as the input of a conversion."""

    ONNX = "ONNX"
    IR = "IR"
    TFLITE = "TFLITE"
    DLC = "DLC"
    HAR = "HAR"
    PYTORCH = "PYTORCH"

    @classmethod
    def from_path(cls, path: str | Path) -> "InputFileType":
        """Determine the input file type from a path.

        Args:
            path: Path to the model file. Only its suffix is examined;
                the file need not exist.

        Returns:
            The matching input file type.

        Raises:
            ValueError: If the suffix belongs to no known format.

        Example:
            >>> InputFileType.from_path("models/yolov6n.onnx")
            <InputFileType.ONNX: 'ONNX'>
            >>> InputFileType.from_path("m.tflite")
            <InputFileType.TFLITE: 'TFLITE'>

        """
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
        if path.suffix in [".pt", ".pth"]:
            return cls.PYTORCH
        raise ValueError(f"Unsupported file type: `{path}`")
