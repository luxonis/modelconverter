"""Factories for building tiny dummy ONNX models on the fly.

These let the host-side unit tests exercise the conversion config /
metadata / nn-archive code paths without downloading real models. ONNX's
``checker.check_model`` does not run shape inference by default, so the
declared input/output ``value_info`` shapes are taken at face value by
``get_metadata`` even when the connecting nodes would produce a different
shape -- this is what makes arbitrary tiny graphs usable as stand-ins.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper

__all__ = [
    "build_hwc_onnx",
    "build_ncd_onnx",
    "build_onnx",
    "build_toy_aggregator_onnx",
    "build_toy_conv_onnx",
    "build_toy_integration_onnx",
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


def build_toy_integration_onnx(
    path: str | Path, *, size: int = 64, with_flag: bool = False
) -> Path:
    """A tiny multi-input network exercising every conversion feature.

    Inputs (each meant to be driven with different mean/scale + encoding
    in the config so the conversion exercises the full preprocessing):

      * ``bgr``  -- FLOAT ``[1, 3, size, size]`` colour input, no channel
        reversal (config ``BGR`` -> ``BGR``);
      * ``rgb``  -- FLOAT ``[1, 3, size, size]`` colour input that *does*
        need reversal (config ``RGB`` -> ``BGR``);
      * ``gray`` -- FLOAT ``[1, 1, size, size]`` single-channel input;
      * ``flag`` -- INT64 ``[1]`` scalar control input (only when
        ``with_flag``), meant to be frozen to a constant at conversion time
        (the input-freezing path). Freezing is an OpenVINO-MO-only feature
        and the resulting rank-1 input is rejected by e.g. the Hailo parser,
        so the shared net omits it and a dedicated RVC2-only test opts in.

    The graph normalizes nothing itself (that is what the baked
    preprocessing is for) -- it just combines the inputs with basic math:
    ``(bgr + rgb + gray)`` (``* flag`` when present), plus a channel-wise
    average, giving two outputs of different rank to exercise multi-output
    handling:

      * ``output`` -- FLOAT ``[1, 3, size, size]``;
      * ``pooled`` -- FLOAT ``[1, 3, 1, 1]``.
    """
    bgr = helper.make_tensor_value_info(
        "bgr", TensorProto.FLOAT, [1, 3, size, size]
    )
    rgb = helper.make_tensor_value_info(
        "rgb", TensorProto.FLOAT, [1, 3, size, size]
    )
    gray = helper.make_tensor_value_info(
        "gray", TensorProto.FLOAT, [1, 1, size, size]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, size, size]
    )
    pooled = helper.make_tensor_value_info(
        "pooled", TensorProto.FLOAT, [1, 3, 1, 1]
    )

    inputs = [bgr, rgb, gray]
    initializers = []
    # (bgr + rgb + gray) -- gray broadcasts over the 3 channels
    nodes = [helper.make_node("Add", ["bgr", "rgb"], ["sum_rb"])]

    if with_flag:
        flag = helper.make_tensor_value_info("flag", TensorProto.INT64, [1])
        inputs.append(flag)
        initializers.append(
            helper.make_tensor(
                name="flag_shape",
                data_type=TensorProto.INT64,
                dims=[4],
                vals=[1, 1, 1, 1],
            )
        )
        nodes += [
            helper.make_node("Add", ["sum_rb", "gray"], ["sum_all"]),
            # flag (INT [1]) -> float scalar broadcastable over NCHW, * sum
            helper.make_node(
                "Cast", ["flag"], ["flag_f"], to=TensorProto.FLOAT
            ),
            helper.make_node("Reshape", ["flag_f", "flag_shape"], ["flag_4d"]),
            helper.make_node("Mul", ["sum_all", "flag_4d"], ["output"]),
        ]
    else:
        nodes.append(helper.make_node("Add", ["sum_rb", "gray"], ["output"]))

    # channel-wise spatial average -> second output of different rank
    nodes.append(
        helper.make_node(
            "ReduceMean", ["output"], ["pooled"], axes=[2, 3], keepdims=1
        )
    )

    graph = helper.make_graph(
        nodes,
        "ToyIntegrationModel",
        inputs,
        [output, pooled],
        initializer=initializers,
    )
    # opset 13: ReduceMean takes `axes` as an attribute (not an input).
    model = helper.make_model(
        graph,
        producer_name="DummyModelProducer",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    # Pin a broadly-supported IR version (the onnx lib may default to one
    # newer than the runtime/vendor tools accept).
    model.ir_version = 9
    checker.check_model(model)
    path = Path(path)
    onnx.save(model, str(path))
    return path


def build_toy_aggregator_onnx(path: str | Path, *, size: int = 64) -> Path:
    """A tiny aggregator: the final stage of the toy multistage net.

    Takes two ``[1, 3, size, size]`` inputs (fed at conversion time from two
    upstream toy stages via linked calibration) and computes a simple
    ``from_first * from_second + bias`` -> ``[1, 3, size, size]``. Inputs are
    raw (no preprocessing), so the converted model has no baked normalization.
    """
    from_first = helper.make_tensor_value_info(
        "from_first", TensorProto.FLOAT, [1, 3, size, size]
    )
    from_second = helper.make_tensor_value_info(
        "from_second", TensorProto.FLOAT, [1, 3, size, size]
    )
    out = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, [1, 3, size, size]
    )
    bias = helper.make_tensor(
        name="bias",
        data_type=TensorProto.FLOAT,
        dims=[1, 3, 1, 1],
        vals=[1.0, 2.0, 3.0],
    )
    nodes = [
        helper.make_node("Mul", ["from_first", "from_second"], ["prod"]),
        helper.make_node("Add", ["prod", "bias"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "ToyAggregatorModel",
        [from_first, from_second],
        [out],
        initializer=[bias],
    )
    model = helper.make_model(
        graph,
        producer_name="DummyModelProducer",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    checker.check_model(model)
    path = Path(path)
    onnx.save(model, str(path))
    return path


def build_toy_conv_onnx(
    path: str | Path,
    *,
    size: int = 32,
    out_channels: int = 4,
    external_data: bool = False,
) -> Path:
    """A tiny single-input conv net for numeric-fidelity (precision) tests.

    Unlike ``build_toy_integration_onnx`` (whose output is essentially the
    preprocessed input, and whose per-input scale spread makes it impossible
    for the Hailo quantizer to fit), this has a real ``Conv`` with modest,
    well-conditioned weights -- so every backend, Hailo included, can quantize
    it and reproduce the fp32 reference near-losslessly.

    A single ``[1, 3, size, size]`` colour input, deterministic 3x3 ``Conv``
    (``same`` padding) -> ``[1, out_channels, size, size]`` output. The input
    still carries mean/scale (+ optional channel reversal) in the config, so
    the converted model exercises the full baked-preprocessing path.

    With ``external_data=True`` the weights are written to a sibling
    ``<name>_data`` file instead of being embedded in the ``.onnx`` (the
    layout ``onnx.save(save_as_external_data=True)`` / modelconverter's
    ``save_onnx_model`` produce), so a conversion exercises the external-data
    copy/preserve branches. ``size_threshold=0`` forces even the tiny conv
    weight out to the sibling file.
    """
    inp = helper.make_tensor_value_info(
        "img", TensorProto.FLOAT, [1, 3, size, size]
    )
    out = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, [1, out_channels, size, size]
    )
    # Deterministic small weights: quantizes cleanly, no host randomness.
    rng = np.random.default_rng(0)
    weight_values = (
        rng.standard_normal((out_channels, 3, 3, 3)) * 0.1
    ).astype(np.float32)
    # ONNX only moves a tensor's ``raw_data`` to the external file, so the
    # weight must be stored raw (not in ``float_data``) for external_data.
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [out_channels, 3, 3, 3],
        weight_values.tobytes() if external_data else weight_values.ravel(),
        raw=external_data,
    )
    node = helper.make_node(
        "Conv",
        ["img", "weight"],
        ["out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph(
        [node], "ToyConvModel", [inp], [out], initializer=[weight]
    )
    model = helper.make_model(
        graph,
        producer_name="DummyModelProducer",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    checker.check_model(model)
    path = Path(path)
    if external_data:
        onnx.save(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{path.name}_data",
            size_threshold=0,
            convert_attribute=False,
        )
    else:
        onnx.save(model, str(path))
    return path


def build_hwc_onnx(
    path: str | Path, *, size: int = 32, channels: int = 3
) -> Path:
    """A tiny model with a single rank-3 ``HWC`` input (no batch dim).

    A shape-preserving ``Relu`` keeps it a real op while leaving the
    ``[H, W, C]`` input untouched -- used to exercise conversion of a 3D
    channels-last input.
    """
    inp = helper.make_tensor_value_info(
        "data", TensorProto.FLOAT, [size, size, channels]
    )
    out = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, [size, size, channels]
    )
    node = helper.make_node("Relu", ["data"], ["out"])
    graph = helper.make_graph([node], "HWCModel", [inp], [out])
    model = helper.make_model(
        graph,
        producer_name="DummyModelProducer",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    checker.check_model(model)
    path = Path(path)
    onnx.save(model, str(path))
    return path


def build_ncd_onnx(
    path: str | Path, *, shape: tuple[int, int, int] = (1, 4, 5)
) -> Path:
    """A tiny model whose sole input is a rank-3 tensor with a leading batch
    dim (default ``[1, 4, 5]``).

    ``make_default_layout`` can't match such a shape to a spatial layout, so it
    falls back to the generic lettercode ``NCD`` -- which is exactly what
    SNPE's ``NCD``/``NDC``/``D`` -> ``F`` layout rewrite on the RVC4 path
    expects. A shape-preserving ``Relu`` keeps it a real op.
    """
    inp = helper.make_tensor_value_info("data", TensorProto.FLOAT, list(shape))
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))
    node = helper.make_node("Relu", ["data"], ["out"])
    graph = helper.make_graph([node], "NCDModel", [inp], [out])
    model = helper.make_model(
        graph,
        producer_name="DummyModelProducer",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
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
