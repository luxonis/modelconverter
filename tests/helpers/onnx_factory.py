"""Factories for tiny dummy ONNX models, built on the fly.

``checker.check_model`` does not run shape inference, so the declared
``value_info`` shapes are taken at face value by ``get_metadata`` even when the
connecting nodes would produce something else. That is what makes arbitrary
tiny graphs usable as stand-ins for real models.
"""

from pathlib import Path

import numpy as np
import onnx
from luxonis_ml.typing import PathType
from onnx import TensorProto, checker, helper, numpy_helper

# Opset 13 keeps `axes` an attribute of ReduceMean rather than an input, and
# IR 9 is accepted by every vendor tool (the onnx lib defaults higher).
_OPSET = 13
_IR_VERSION = 9


def _save(
    graph: onnx.GraphProto, path: PathType, *, external_data: bool = False
) -> Path:
    model = helper.make_model(
        graph,
        producer_name="DummyModelProducer",
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    model.ir_version = _IR_VERSION
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


def build_onnx(
    path: PathType,
    inputs: list[tuple[str, list[int], int]],
    outputs: list[tuple[str, list[int], int]],
) -> Path:
    """A minimal valid model with the given inputs/outputs.

    Each output comes off an ``Identity`` node fed from an input, so the graph
    is structurally valid whatever the declared shapes.
    """
    graph_inputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, shape, dtype in inputs
    ]
    graph_outputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, shape, dtype in outputs
    ]
    nodes = [
        helper.make_node(
            "Identity",
            inputs=[inputs[min(i, len(inputs) - 1)][0]],
            outputs=[out_name],
        )
        for i, (out_name, _, _) in enumerate(outputs)
    ]
    graph = helper.make_graph(nodes, "DummyModel", graph_inputs, graph_outputs)
    return _save(graph, path)


def build_toy_integration_onnx(
    path: PathType, *, size: int = 64, with_flag: bool = False
) -> Path:
    """A tiny multi-input net covering the whole preprocessing surface.

    Three image inputs, each meant to carry different mean/scale + encoding in
    the config: ``bgr`` (``BGR`` -> ``BGR``, no reversal), ``rgb`` (``RGB`` ->
    ``BGR``, reversed) and single-channel ``gray``. ``with_flag`` adds an INT64
    scalar ``flag`` and a 3-element ``chan_scale``, meant to be frozen to a
    scalar and to a list constant -- both ``frozen_value`` branches. Freezing is
    OpenVINO-MO only and other parsers reject the resulting rank-1 inputs, so
    the shared net leaves them out and one RVC2 test opts in.

    The graph normalizes nothing itself (that is what the baked preprocessing is
    for): ``bgr + rgb + gray`` (``* flag * chan_scale`` when present) gives
    ``output`` ``[1, 3, size, size]``, and a channel-wise spatial average gives
    ``pooled`` ``[1, 3, 1, 1]`` -- two outputs of different rank.
    """

    def image(name: str, channels: int) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(
            name, TensorProto.FLOAT, [1, channels, size, size]
        )

    inputs = [image("bgr", 3), image("rgb", 3), image("gray", 1)]
    outputs = [
        image("output", 3),
        helper.make_tensor_value_info(
            "pooled", TensorProto.FLOAT, [1, 3, 1, 1]
        ),
    ]
    initializers = []
    # gray broadcasts over the 3 channels
    nodes = [helper.make_node("Add", ["bgr", "rgb"], ["sum_rb"])]

    if with_flag:
        inputs += [
            helper.make_tensor_value_info("flag", TensorProto.INT64, [1]),
            helper.make_tensor_value_info(
                "chan_scale", TensorProto.FLOAT, [3]
            ),
        ]
        initializers += [
            helper.make_tensor("flag_shape", TensorProto.INT64, [4], [1] * 4),
            helper.make_tensor(
                "chan_scale_shape", TensorProto.INT64, [4], [1, 3, 1, 1]
            ),
        ]
        nodes += [
            helper.make_node("Add", ["sum_rb", "gray"], ["sum_all"]),
            # flag (INT [1]) -> float scalar broadcastable over NCHW
            helper.make_node(
                "Cast", ["flag"], ["flag_f"], to=TensorProto.FLOAT
            ),
            helper.make_node("Reshape", ["flag_f", "flag_shape"], ["flag_4d"]),
            helper.make_node("Mul", ["sum_all", "flag_4d"], ["mul_flag"]),
            # chan_scale (FLOAT [3]) -> per-channel scale over NCHW
            helper.make_node(
                "Reshape",
                ["chan_scale", "chan_scale_shape"],
                ["chan_scale_4d"],
            ),
            helper.make_node("Mul", ["mul_flag", "chan_scale_4d"], ["output"]),
        ]
    else:
        nodes.append(helper.make_node("Add", ["sum_rb", "gray"], ["output"]))

    nodes.append(
        helper.make_node(
            "ReduceMean", ["output"], ["pooled"], axes=[2, 3], keepdims=1
        )
    )

    graph = helper.make_graph(
        nodes,
        "ToyIntegrationModel",
        inputs,
        outputs,
        initializer=initializers,
    )
    return _save(graph, path)


def build_toy_aggregator_onnx(path: PathType, *, size: int = 64) -> Path:
    """The final stage of the toy multistage net.

    Two ``[1, 3, size, size]`` inputs -- fed from two upstream toy stages via
    linked calibration -- combined as ``from_first * from_second + bias``. The
    inputs are raw, so the converted model has no baked normalization.
    """
    shape = [1, 3, size, size]
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name in ("from_first", "from_second")
    ]
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)
    bias = helper.make_tensor(
        "bias", TensorProto.FLOAT, [1, 3, 1, 1], [1.0, 2.0, 3.0]
    )
    nodes = [
        helper.make_node("Mul", ["from_first", "from_second"], ["prod"]),
        helper.make_node("Add", ["prod", "bias"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes, "ToyAggregatorModel", inputs, [out], initializer=[bias]
    )
    return _save(graph, path)


def build_toy_conv_onnx(
    path: PathType,
    *,
    size: int = 32,
    out_channels: int = 4,
    external_data: bool = False,
) -> Path:
    """A tiny single-input conv net for the numeric-fidelity tests.

    ``build_toy_integration_onnx`` is a poor fidelity subject -- its output is
    essentially the preprocessed input, and its per-input scale spread is too
    wide for the Hailo quantizer to fit. This has a real 3x3 ``Conv`` with
    modest, well-conditioned weights instead, which every backend can quantize
    and reproduce near-losslessly. The input still carries mean/scale (+ channel
    reversal) in the config, so the full baked-preprocessing path is exercised.

    ``external_data=True`` writes the weights to a sibling ``<name>_data`` file
    instead of embedding them, so a conversion exercises the external-data
    copy/preserve branches.
    """
    inp = helper.make_tensor_value_info(
        "img", TensorProto.FLOAT, [1, 3, size, size]
    )
    out = helper.make_tensor_value_info(
        "out", TensorProto.FLOAT, [1, out_channels, size, size]
    )
    rng = np.random.default_rng(0)
    values = (rng.standard_normal((out_channels, 3, 3, 3)) * 0.1).astype(
        np.float32
    )
    # ONNX only externalizes a tensor's `raw_data`, so for external_data the
    # weight has to be stored raw rather than in `float_data`.
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [out_channels, 3, 3, 3],
        values.tobytes() if external_data else values.ravel(),
        raw=external_data,
    )
    node = helper.make_node(
        "Conv", ["img", "weight"], ["out"], kernel_shape=[3, 3], pads=[1] * 4
    )
    graph = helper.make_graph(
        [node], "ToyConvModel", [inp], [out], initializer=[weight]
    )
    return _save(graph, path, external_data=external_data)


def weights_only_external_onnx(
    path: PathType, *, single_file: bool = True
) -> list[Path]:
    """A model that is nothing but externally stored weights.

    Returns the companion files, in sorted order. Anything walking a
    model's external data has to find every one of them, so
    ``single_file=False`` stores one tensor per file -- what
    ``all_tensors_to_one_file=False`` produces -- and the default packs
    them into the single sibling ``<name>_data`` that a real large model
    uses.

    The graph declares no inputs or outputs: these are subjects for the
    file bookkeeping around a model, never for a conversion, and leaving
    the graph empty keeps what each test is about in the test.
    """
    path = Path(path)
    tensors = [
        numpy_helper.from_array(
            np.asarray([1.0, 2.0], dtype=np.float32), name="first"
        ),
        numpy_helper.from_array(
            np.asarray([3.0, 4.0], dtype=np.float32), name="second"
        ),
    ]
    if single_file:
        tensors = tensors[:1]
    model = helper.make_model(
        helper.make_graph([], "ExternalWeights", [], [], tensors)
    )
    # The companion files are whatever the save newly produced. Diffing the
    # directory rather than listing it afterwards keeps unrelated entries --
    # the re-rooted cache lives under the same tmp_path -- out of the result.
    before = set(path.parent.iterdir())
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=single_file,
        # One file per tensor is only reached by leaving the location
        # unset; naming it would collapse them back into that one file.
        location=f"{path.name}_data" if single_file else None,
        # Without this the tiny tensors stay embedded and no companion
        # file is written at all.
        size_threshold=0,
    )
    return sorted(set(path.parent.iterdir()) - before - {path})


def build_relu_onnx(path: PathType, shape: list[int]) -> Path:
    """A shape-preserving ``Relu`` over a single input of the given shape.

    Used for conversions of shapes the usual 4D ``NCHW`` models never produce:
    a rank-3 ``HWC`` (channels-last, no batch dim) input, or a rank-3
    ``[1, C, D]`` one, which ``make_default_layout`` can only label ``NCD`` --
    exactly what SNPE's ``NCD``/``NDC``/``D`` -> ``F`` rewrite expects.
    """
    inp = helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)
    node = helper.make_node("Relu", ["data"], ["out"])
    return _save(helper.make_graph([node], "ReluModel", [inp], [out]), path)


def split_concat_onnx(path: PathType) -> Path:
    """A ``Split`` whose two branches a ``Concat`` rejoins.

    The Concat produces a graph output, so nothing consumes it. The
    Split-Concat fusion walks forward from the Concat to find a ``Conv``,
    and this graph gives that walk nowhere to go.
    """
    inp = helper.make_tensor_value_info("img", TensorProto.FLOAT, [1, 4, 8, 8])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 8, 8])
    nodes = [
        helper.make_node("Split", ["img"], ["low", "high"], axis=1),
        helper.make_node("Concat", ["low", "high"], ["out"], axis=1),
    ]
    graph = helper.make_graph(nodes, "SplitConcatModel", [inp], [out])
    return _save(graph, path)


def standard_dummy_onnx(path: PathType) -> Path:
    """The canonical two-in/two-out model used across the config tests."""
    inputs = [
        helper.make_tensor_value_info(
            "input0", TensorProto.FLOAT, [1, 3, 64, 64]
        ),
        helper.make_tensor_value_info(
            "input1", TensorProto.FLOAT, [1, 3, 128, 128]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 10]),
        helper.make_tensor_value_info(
            "output1", TensorProto.FLOAT, [1, 5, 5, 5]
        ),
    ]
    shape_tensor = helper.make_tensor(
        "shape_tensor", TensorProto.INT64, [4], [1, 5, 5, 5]
    )
    nodes = [
        helper.make_node("Add", ["input0", "input0"], ["doubled0"]),
        helper.make_node("Add", ["input1", "input1"], ["doubled1"]),
        helper.make_node("Flatten", ["doubled0"], ["output0"]),
        helper.make_node("Reshape", ["doubled1", "shape_tensor"], ["output1"]),
    ]
    graph = helper.make_graph(
        nodes, "DummyModel", inputs, outputs, initializer=[shape_tensor]
    )
    return _save(graph, path)


def single_io_onnx(
    path: PathType,
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


def grayscale_onnx(path: PathType) -> Path:
    """Single grayscale (1-channel) input model."""
    return single_io_onnx(path, shape=[1, 1, 64, 64])


def dynamic_batch_onnx(path: PathType) -> Path:
    """Model whose input batch dimension is 0 (dynamic)."""
    return single_io_onnx(path, shape=[0, 3, 64, 64])


def intermediate_info_onnx(path: PathType) -> Path:
    """Model with one intermediate tensor described in ``value_info`` and one
    that is not, for the ONNX introspection helpers in config.py."""
    inp = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    out = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    described = helper.make_tensor_value_info(
        "described_node", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    nodes = [
        helper.make_node(
            "Relu", ["input0"], ["described_node"], name="described_node"
        ),
        helper.make_node(
            "Relu",
            ["described_node"],
            ["undescribed_node"],
            name="undescribed_node",
        ),
        helper.make_node("Identity", ["undescribed_node"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "IntermediateModel", [inp], [out], value_info=[described]
    )
    return _save(graph, path)


def multi_file_external_onnx(path: PathType) -> list[Path]:
    """A conv model whose weight and bias each live in their own file.

    Returns the companion files, in sorted order. This is the layout
    ``all_tensors_to_one_file=False`` produces; two initializers are the
    smallest model that tells "copies the external data" apart from
    "copies *all* of it". Unlike L{weights_only_external_onnx} the graph
    has real inputs and outputs, so it can be put through a conversion.
    """
    path = Path(path)
    inp = helper.make_tensor_value_info("img", TensorProto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 8, 8])
    node = helper.make_node(
        "Conv",
        ["img", "weight", "bias"],
        ["out"],
        kernel_shape=[3, 3],
        pads=[1] * 4,
    )
    graph = helper.make_graph(
        [node],
        "MultiFileExternal",
        [inp],
        [out],
        initializer=[
            numpy_helper.from_array(
                np.zeros((2, 3, 3, 3), dtype=np.float32), name="weight"
            ),
            numpy_helper.from_array(
                np.zeros((2,), dtype=np.float32), name="bias"
            ),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", _OPSET)]
    )
    model.ir_version = _IR_VERSION
    checker.check_model(model)

    before = set(path.parent.iterdir())
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        # Naming a location would collapse the tensors back into one file.
        all_tensors_to_one_file=False,
        size_threshold=0,
    )
    return sorted(set(path.parent.iterdir()) - before - {path})
