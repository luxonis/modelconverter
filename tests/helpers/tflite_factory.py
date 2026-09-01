"""Factory for a hand-built minimal ``.tflite`` model.

Building a TFLite model usually means pulling in TensorFlow. A TFLite file is
just a FlatBuffer, though, so for a tiny fixed graph we can emit the bytes
directly with ``flatbuffers`` (a light dependency) against the frozen TFLite
schema -- no TensorFlow needed. The result is a single-op ``Relu`` network
that ``tflite2onnx`` (the converter modelconverter's RVC2 path uses) accepts.
"""

from pathlib import Path

import flatbuffers
from luxonis_ml.typing import PathType

# TFLite schema enum values (frozen for backward compatibility).
_RELU = 19  # BuiltinOperator.RELU
_FLOAT32 = 0  # TensorType.FLOAT32
_TFLITE_SCHEMA_VERSION = 3


def _int_vector(b: flatbuffers.Builder, values: list[int]) -> int:
    b.StartVector(4, len(values), 4)
    for v in reversed(values):
        b.PrependInt32(v)
    return b.EndVector()


def build_toy_tflite(
    path: PathType, *, shape: tuple[int, ...] = (1, 8, 8, 3)
) -> Path:
    """Write a minimal ``input -> Relu -> output`` TFLite model.

    A single elementwise op keeps the graph trivially convertible; the NHWC
    ``shape`` (default ``[1, 8, 8, 3]``) is carried on both tensors.
    """
    b = flatbuffers.Builder(1024)
    in_name = b.CreateString("input")
    out_name = b.CreateString("output")
    description = b.CreateString("toy")
    subgraph_name = b.CreateString("main")

    # OperatorCode -- set both the legacy byte code and the int32 code; TFLite
    # keeps them in sync for op codes below 127 and readers may use either.
    b.StartObject(4)
    b.PrependInt8Slot(0, _RELU, 0)  # deprecated_builtin_code
    b.PrependInt32Slot(3, _RELU, 0)  # builtin_code
    operator_code = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(operator_code)
    operator_codes = b.EndVector()

    # One empty buffer (index 0) shared by the data-less activation tensors.
    b.StartObject(1)
    empty_buffer = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(empty_buffer)
    buffers = b.EndVector()

    def tensor(name: int) -> int:
        shape_vec = _int_vector(b, list(shape))
        b.StartObject(4)
        b.PrependUOffsetTRelativeSlot(0, shape_vec, 0)
        b.PrependInt8Slot(1, _FLOAT32, 0)
        b.PrependUint32Slot(2, 0, 0)  # buffer 0 (no data)
        b.PrependUOffsetTRelativeSlot(3, name, 0)
        return b.EndObject()

    input_tensor = tensor(in_name)
    output_tensor = tensor(out_name)
    b.StartVector(4, 2, 4)
    b.PrependUOffsetTRelative(output_tensor)
    b.PrependUOffsetTRelative(input_tensor)
    tensors = b.EndVector()

    op_inputs = _int_vector(b, [0])
    op_outputs = _int_vector(b, [1])
    b.StartObject(3)
    b.PrependUint32Slot(0, 0, 0)  # opcode_index
    b.PrependUOffsetTRelativeSlot(1, op_inputs, 0)
    b.PrependUOffsetTRelativeSlot(2, op_outputs, 0)
    operator = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(operator)
    operators = b.EndVector()

    sg_inputs = _int_vector(b, [0])
    sg_outputs = _int_vector(b, [1])
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(0, tensors, 0)
    b.PrependUOffsetTRelativeSlot(1, sg_inputs, 0)
    b.PrependUOffsetTRelativeSlot(2, sg_outputs, 0)
    b.PrependUOffsetTRelativeSlot(3, operators, 0)
    b.PrependUOffsetTRelativeSlot(4, subgraph_name, 0)
    subgraph = b.EndObject()
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(subgraph)
    subgraphs = b.EndVector()

    b.StartObject(8)
    b.PrependUint32Slot(0, _TFLITE_SCHEMA_VERSION, 0)
    b.PrependUOffsetTRelativeSlot(1, operator_codes, 0)
    b.PrependUOffsetTRelativeSlot(2, subgraphs, 0)
    b.PrependUOffsetTRelativeSlot(3, description, 0)
    b.PrependUOffsetTRelativeSlot(4, buffers, 0)
    model = b.EndObject()

    b.Finish(model, file_identifier=b"TFL3")
    path = Path(path)
    path.write_bytes(bytes(b.Output()))
    return path
