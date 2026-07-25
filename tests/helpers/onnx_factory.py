"""Factories for building tiny dummy ONNX models on the fly.

These let the host-side unit tests exercise the conversion config /
metadata / nn-archive code paths without downloading real models. ONNX's
``checker.check_model`` does not run shape inference by default, so the
declared input/output ``value_info`` shapes are taken at face value by
``get_metadata`` even when the connecting nodes would produce a different
shape -- this is what makes arbitrary tiny graphs usable as stand-ins.
"""

from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper

__all__ = [
    "build_onnx",
    "dynamic_batch_onnx",
    "grayscale_onnx",
    "intermediate_info_onnx",
    "single_io_onnx",
    "standard_dummy_onnx",
]


def build_onnx(
    path: str | Path,
    inputs: list[tuple[str, list[int], int]],
    outputs: list[tuple[str, list[int], int]],
    *,
    producer: str = "DummyModelProducer",
) -> Path:
    """Builds a minimal valid ONNX model with the given inputs/outputs.

    Each output is produced by an ``Identity`` node fed from an input,
    so the model is structurally valid regardless of the declared
    shapes.
    """
    graph_inputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, shape, dtype in inputs
    ]
    graph_outputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, shape, dtype in outputs
    ]
    nodes = []
    for i, (out_name, _, _) in enumerate(outputs):
        src_name = inputs[min(i, len(inputs) - 1)][0]
        nodes.append(
            helper.make_node("Identity", inputs=[src_name], outputs=[out_name])
        )

    graph = helper.make_graph(nodes, "DummyModel", graph_inputs, graph_outputs)
    model = helper.make_model(graph, producer_name=producer)
    checker.check_model(model)
    path = Path(path)
    onnx.save(model, str(path))
    return path


def standard_dummy_onnx(path: str | Path) -> Path:
    """The canonical two-input / two-output model used across the config
    tests (Add + Flatten + Reshape with a shape initializer)."""
    input0 = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    input1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, [1, 3, 128, 128]
    )
    output0 = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 10]
    )
    output1 = helper.make_tensor_value_info(
        "output1", TensorProto.FLOAT, [1, 5, 5, 5]
    )
    shape_tensor = helper.make_tensor(
        name="shape_tensor",
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[1, 5, 5, 5],
    )
    node0 = helper.make_node(
        "Add", inputs=["input0", "input0"], outputs=["intermediate0"]
    )
    node1 = helper.make_node(
        "Add", inputs=["input1", "input1"], outputs=["intermediate1"]
    )
    node2 = helper.make_node(
        "Flatten", inputs=["intermediate0"], outputs=["output0"]
    )
    node3 = helper.make_node(
        "Reshape",
        inputs=["intermediate1", "shape_tensor"],
        outputs=["output1"],
    )
    graph = helper.make_graph(
        [node0, node1, node2, node3],
        "DummyModel",
        [input0, input1],
        [output0, output1],
        initializer=[shape_tensor],
    )
    model = helper.make_model(graph, producer_name="DummyModelProducer")
    checker.check_model(model)
    path = Path(path)
    onnx.save(model, str(path))
    return path


def single_io_onnx(
    path: str | Path,
    *,
    name: str = "input0",
    shape: list[int] | None = None,
    dtype: int = TensorProto.FLOAT,
    output_name: str = "output0",
    output_shape: list[int] | None = None,
) -> Path:
    """A single-input / single-output model."""
    return build_onnx(
        path,
        inputs=[(name, shape or [1, 3, 64, 64], dtype)],
        outputs=[(output_name, output_shape or [1, 10], dtype)],
    )


def grayscale_onnx(path: str | Path) -> Path:
    """Single grayscale (1-channel) input model."""
    return single_io_onnx(path, shape=[1, 1, 64, 64])


def dynamic_batch_onnx(path: str | Path) -> Path:
    """Model whose input batch dimension is 0 (dynamic)."""
    return single_io_onnx(path, shape=[0, 3, 64, 64])


def intermediate_info_onnx(path: str | Path) -> Path:
    """Model exposing an intermediate tensor in ``value_info`` with a
    static shape, plus one node whose output is not described -- used to
    exercise the ONNX node/tensor introspection helpers in config.py."""
    input0 = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    output0 = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    described = helper.make_tensor_value_info(
        "described_node", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    node0 = helper.make_node(
        "Relu",
        inputs=["input0"],
        outputs=["described_node"],
        name="described_node",
    )
    node1 = helper.make_node(
        "Relu",
        inputs=["described_node"],
        outputs=["undescribed_node"],
        name="undescribed_node",
    )
    node2 = helper.make_node(
        "Identity", inputs=["undescribed_node"], outputs=["output0"]
    )
    graph = helper.make_graph(
        [node0, node1, node2],
        "IntermediateModel",
        [input0],
        [output0],
        value_info=[described],
    )
    model = helper.make_model(graph, producer_name="DummyModelProducer")
    checker.check_model(model)
    path = Path(path)
    onnx.save(model, str(path))
    return path
