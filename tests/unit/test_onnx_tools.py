"""Host-side unit tests for the ONNX graph rewrites.

The normalization tests use tiny identity models so graph structure and
numerical behavior can both be checked without a vendor toolchain.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto

from modelconverter.utils.config import InputConfig
from modelconverter.utils.exceptions import (
    ONNXException,
    PreprocessingEmbeddingError,
)
from modelconverter.utils.onnx_tools import (
    ONNXModifier,
    onnx_attach_normalization_to_inputs,
)
from tests.helpers.onnx_factory import single_io_onnx, split_concat_onnx


def test_split_concat_fusion_leaves_a_terminal_concat_alone(tmp_path: Path):
    """A Concat that feeds nothing ends the forward walk for a ``Conv``.

    The walk recorded the missing successor before it tested for it, so
    the search then read ``op`` off ``None``. Every model whose Concat
    produced a graph output raised ``AttributeError`` here, which took
    down the whole conversion.
    """
    modifier = ONNXModifier(
        split_concat_onnx(tmp_path / "model.onnx"),
        tmp_path / "modified.onnx",
        skip_optimization=True,
        skip_constant_folding=True,
    )

    modifier._fuse_split_concat_to_conv()

    assert [node.op for node in modifier._onnx_gs.nodes] == [
        "Split",
        "Concat",
    ]


@pytest.mark.parametrize(
    ("shape", "layout", "values_shape"),
    [
        ([1, 2, 3, 4], "NCHW", (1, 2, 1, 1)),
        ([1, 3, 4, 2], "NHWC", (1, 1, 1, 2)),
    ],
)
def test_two_channel_normalization_is_embedded_and_numerically_correct(
    tmp_path: Path,
    shape: list[int],
    layout: str,
    values_shape: tuple[int, ...],
):
    model_path = single_io_onnx(
        tmp_path / f"{layout}.onnx",
        shape=shape,
        output_shape=shape,
    )
    config = InputConfig(
        name="input0",
        shape=shape,
        layout=layout,
        encoding="NONE",
        mean_values=[10.0, 20.0],
        scale_values=[2.0, 4.0],
    )

    modified = onnx_attach_normalization_to_inputs(
        model_path,
        tmp_path / f"{layout}-modified.onnx",
        {"input0": config},
    )

    graph = onnx.load(modified).graph
    assert [node.op_type for node in graph.node[:2]] == ["Sub", "Mul"]
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = (x - np.array([10.0, 20.0]).reshape(values_shape)) / np.array(
        [2.0, 4.0]
    ).reshape(values_shape)
    actual = ort.InferenceSession(str(modified)).run(None, {"input0": x})[0]
    np.testing.assert_allclose(actual, expected)


def test_scalar_normalization_broadcasts_to_every_channel(tmp_path: Path):
    shape = [1, 2, 3, 4]
    model_path = single_io_onnx(
        tmp_path / "scalar.onnx", shape=shape, output_shape=shape
    )
    config = InputConfig(
        name="input0",
        shape=shape,
        layout="NCHW",
        encoding="NONE",
        mean_values=3.0,
        scale_values=2.0,
    )

    modified = onnx_attach_normalization_to_inputs(
        model_path,
        tmp_path / "scalar-modified.onnx",
        {"input0": config},
    )

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    actual = ort.InferenceSession(str(modified)).run(None, {"input0": x})[0]
    np.testing.assert_allclose(actual, (x - 3.0) / 2.0)


def test_invalid_normalization_value_count_fails(tmp_path: Path):
    shape = [1, 2, 3, 4]
    model_path = single_io_onnx(
        tmp_path / "count.onnx", shape=shape, output_shape=shape
    )
    config = InputConfig(
        name="input0",
        shape=shape,
        layout="NCHW",
        encoding="NONE",
        mean_values=[1.0, 2.0, 3.0],
    )

    with pytest.raises(ONNXException, match="one value per channel"):
        onnx_attach_normalization_to_inputs(
            model_path,
            tmp_path / "count-modified.onnx",
            {"input0": config},
        )


def test_non_three_channel_color_reversal_fails(tmp_path: Path):
    shape = [1, 2, 3, 4]
    model_path = single_io_onnx(
        tmp_path / "reverse.onnx", shape=shape, output_shape=shape
    )
    config = InputConfig(
        name="input0",
        shape=shape,
        layout="NCHW",
        encoding={"from": "RGB", "to": "BGR"},
    )

    with pytest.raises(ONNXException, match="requires exactly 3 channels"):
        onnx_attach_normalization_to_inputs(
            model_path,
            tmp_path / "reverse-modified.onnx",
            {"input0": config},
        )


def test_unsupported_layout_with_preprocessing_fails(tmp_path: Path):
    shape = [1, 2]
    model_path = single_io_onnx(
        tmp_path / "layout.onnx", shape=shape, output_shape=shape
    )
    config = InputConfig(
        name="input0",
        shape=shape,
        layout="NC",
        encoding="NONE",
        mean_values=[1.0, 2.0],
    )

    with pytest.raises(
        PreprocessingEmbeddingError, match="only 'NCHW' and 'NHWC'"
    ):
        onnx_attach_normalization_to_inputs(
            model_path,
            tmp_path / "layout-modified.onnx",
            {"input0": config},
        )


def test_color_normalization_is_correct_without_mutating_config(
    tmp_path: Path,
):
    shape = [1, 3, 3, 4]
    model_path = single_io_onnx(
        tmp_path / "color.onnx", shape=shape, output_shape=shape
    )
    config = InputConfig(
        name="input0",
        shape=shape,
        layout="NCHW",
        encoding={"from": "RGB", "to": "BGR"},
        mean_values=[1.0, 2.0, 3.0],
        scale_values=[4.0, 5.0, 6.0],
    )
    original = config.model_dump()

    modified = onnx_attach_normalization_to_inputs(
        model_path,
        tmp_path / "color-modified.onnx",
        {"input0": config},
    )

    assert config.model_dump() == original

    bgr = np.array([30.0, 20.0, 10.0], dtype=np.float32).reshape(1, 3, 1, 1)
    bgr = np.broadcast_to(bgr, shape).copy()
    rgb = bgr[:, ::-1, :, :]
    expected = (
        rgb - np.array([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)
    ) / np.array([4.0, 5.0, 6.0]).reshape(1, 3, 1, 1)
    actual = ort.InferenceSession(str(modified)).run(None, {"input0": bgr})[0]
    np.testing.assert_allclose(actual, expected)


def test_integer_mean_scale_normalization_is_unembeddable(tmp_path: Path):
    shape = [1, 2, 3, 4]
    model_path = single_io_onnx(
        tmp_path / "uint8.onnx",
        shape=shape,
        output_shape=shape,
        dtype=TensorProto.UINT8,
    )
    config = InputConfig(
        name="input0",
        shape=shape,
        layout="NCHW",
        encoding="NONE",
        mean_values=[10.0, 20.0],
        scale_values=[2.0, 4.0],
    )

    with pytest.raises(
        PreprocessingEmbeddingError,
        match="normalization requires a floating-point input",
    ):
        onnx_attach_normalization_to_inputs(
            model_path,
            tmp_path / "uint8-modified.onnx",
            {"input0": config},
        )


def test_integer_channel_reversal_remains_supported(tmp_path: Path):
    shape = [1, 3, 2, 2]
    model_path = single_io_onnx(
        tmp_path / "uint8-reverse.onnx",
        shape=shape,
        output_shape=shape,
        dtype=TensorProto.UINT8,
    )
    config = InputConfig(
        name="input0",
        shape=shape,
        layout="NCHW",
        encoding={"from": "RGB", "to": "BGR"},
    )

    modified = onnx_attach_normalization_to_inputs(
        model_path,
        tmp_path / "uint8-reverse-modified.onnx",
        {"input0": config},
    )

    bgr = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    actual = ort.InferenceSession(str(modified)).run(None, {"input0": bgr})[0]
    np.testing.assert_array_equal(actual, bgr[:, ::-1, :, :])
