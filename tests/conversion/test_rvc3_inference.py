"""CPU validation must preserve VPU quantization without AVX2 saturation."""

from pathlib import Path

import numpy as np
import pytest

from modelconverter.platforms.rvc3.inferer import RVC3Inferer
from modelconverter.utils.types import DataType, Encoding, ResizeMethod


@pytest.mark.rvc3
def test_quantized_convolution_keeps_full_int8_range(tmp_path: Path):
    from openvino.runtime import Model, serialize
    from openvino.runtime import opset8 as ops

    shape = [1, 3, 4, 5]
    inp = ops.parameter(shape, np.float32, name="input")
    low, high = (ops.constant(np.float32(value)) for value in (0, 255))
    quantized = ops.fake_quantize(inp, low, high, low, high, 256)
    weights = ops.constant(np.full((1, 3, 1, 1), 127, np.float32))
    wlow, whigh = (ops.constant(np.float32(value)) for value in (-128, 127))
    weights = ops.fake_quantize(weights, wlow, whigh, wlow, whigh, 256)
    out = ops.convolution(quantized, weights, [1, 1], [0, 0], [0, 0], [1, 1])
    out.set_friendly_name("output")
    out.output(0).get_tensor().set_names({"output"})
    model_path = tmp_path / "quantized.xml"
    serialize(Model([out], [inp]), str(model_path))

    sample = np.linspace(-10.25, 260.25, 20, dtype=np.float32).reshape(
        1, 1, 4, 5
    )
    sample = np.repeat(sample, 3, axis=1)
    np.save(tmp_path / "input.npy", sample)
    inferer = RVC3Inferer(
        model_path=model_path,
        src=tmp_path,
        dest=tmp_path / "results",
        in_shapes={"input": shape},
        in_dtypes={"input": DataType.FLOAT32},
        out_shapes={"output": [1, 1, 4, 5]},
        out_dtypes={"output": DataType.FLOAT32},
        resize_method={"input": ResizeMethod.RESIZE},
        encoding={"input": Encoding.BGR},
        layout={"input": "NCHW"},
    )
    actual = np.asarray(
        inferer.infer({"input": tmp_path / "input.npy"})["output"],
        dtype=np.float32,
    )

    # Clipping and rounding must remain active; an unquantized FP32 graph
    # would pass an overflow-only check but give different values here.
    expected = (
        np.rint(np.clip(sample, 0, 255)).sum(axis=1, keepdims=True) * 127
    )
    np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0)
