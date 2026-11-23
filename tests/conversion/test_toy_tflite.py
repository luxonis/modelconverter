"""Toy TFLite conversion + precision.

A hand-built single-op ``.tflite`` is converted on rvc2/rvc3/rvc4, exercising each
backend's distinct TFLite entry path -- ``tflite2onnx`` (rvc2/rvc3) and
``snpe-tflite-to-dlc`` (rvc4).

The model has no weights and no math beyond a ``Relu``, so the converted model
has to return its input image. The conversion feeds the model RGB, as the TFLite
model expects, so each output pixel must hold the RGB values of the image, within
the quantization step.

Hailo is left out: its quantizer only flakily handles this weightless
pass-through (the ``bgr_to_rgb`` layer degenerates to 0 dB SNR and sometimes
yields NaN params), so a hailo case here would be neither a dependable pass nor a
dependable fail. That would need a weighted model.

Run inside the platform Docker image, e.g.::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k toy_tflite'
"""

from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.platforms.getters import get_inferer
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.preprocessing import reorder_layout
from modelconverter.utils.types import Platform
from tests.helpers.conversion import write_config
from tests.helpers.onnx_factory import build_onnx
from tests.helpers.platform_options import platform_options
from tests.helpers.platforms import platform_params
from tests.helpers.precision import RecordProperty, locate_converted_model
from tests.helpers.tflite_factory import build_toy_tflite

PLATFORMS = ("rvc2", "rvc3", "rvc4")
_HEIGHT = 6
_WIDTH = 8
# One step of 8-bit quantization over the 0-255 calibration range. CI
# measures no error at all.
_TOLERANCE = 1.0
# Three distinct channel values inside the default random-calibration range
# (mean 127.5, std 35). Positive so the Relu is a clean identity, and
# well-separated so the quantizers keep them apart.
_CHANNEL_VALUES = (90, 130, 170)

_PARAMS = platform_params(PLATFORMS)


def _conversion_options(platform: Platform) -> tuple[str, ...]:
    """Use the source TFLite model's RGB contract on RVC4."""
    options = platform_options(platform)
    if platform is Platform.RVC4:
        return (*options, "encoding", "RGB")
    return options


@pytest.fixture(scope="module")
def toy_tflite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("toy_tflite")
    return build_toy_tflite(
        workdir / "toy.tflite", shape=(1, _HEIGHT, _WIDTH, 3)
    )


@pytest.fixture(scope="module")
def color_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Vary channels and spatial coordinates to expose layout mistakes."""
    path = tmp_path_factory.mktemp("toy_tflite_img") / "colors.png"
    rows, columns = np.indices((_HEIGHT, _WIDTH))
    cv2.imwrite(
        str(path),
        (np.asarray(_CHANNEL_VALUES) + (2 * rows + columns)[..., None]).astype(
            np.uint8
        ),
    )
    return path


@pytest.mark.parametrize("platform_name", _PARAMS)
def test_toy_tflite_precision(
    platform_name: str,
    toy_tflite: Path,
    color_image: Path,
    record_property: RecordProperty,
):
    platform = Platform(platform_name)
    options = _conversion_options(platform)
    output_name = f"_toy-tflite-prec-{platform_name}"
    convert(
        platform,
        *options,
        path=str(toy_tflite),
        output_dir=output_name,
        to="native",
    )

    # A bare model path is not a config file -- pass it as an `input_model`
    # override with `path=None`, exactly as `convert` does internally.
    cfg, _, _ = get_configs(
        platform, None, ["input_model", str(toy_tflite), *options]
    )
    stage = next(iter(cfg.stages.values()))
    model_path = locate_converted_model(
        OUTPUTS_DIR / output_name, platform_name
    )
    inferer = get_inferer(
        platform,
        str(model_path),
        color_image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer({stage.inputs[0].name: color_image})

    (output,) = converted.values()
    output_array = np.asarray(output, dtype=np.float32)
    output_config = stage.outputs[0]
    output_layout = (
        "NCHW" if platform is Platform.RVC4 else output_config.layout
    )
    assert output_layout is not None
    pixels = reorder_layout(output_array, output_layout, "NHWC")
    assert pixels.shape == (1, _HEIGHT, _WIDTH, 3)
    # OpenCV wrote the image as BGR, and the model gets it as RGB.
    image = cv2.imread(str(color_image))
    assert image is not None
    expected_rgb = image[None, ..., ::-1].astype(np.float32)
    max_error = float(np.abs(pixels - expected_rgb).max())
    record_property(f"{platform_name} max error", round(max_error, 3))
    assert max_error <= _TOLERANCE, (
        f"{platform_name} tflite output differs from the RGB input by "
        f"{max_error:.3f}: expected channel means "
        f"{expected_rgb.reshape(-1, 3).mean(axis=0).tolist()}, got channel "
        f"means {pixels.reshape(-1, 3).mean(axis=0).tolist()}"
    )


@pytest.mark.rvc2
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("encoding", ["RGB", "BGR"])
def test_openvino_channel_order_with_per_channel_normalization(
    tmp_path: Path, color_image: Path, layout: str, encoding: str
):
    """Channel reversal must also preserve the order of unequal means/scales."""
    shape = (
        [1, 3, _HEIGHT, _WIDTH]
        if layout == "NCHW"
        else [1, _HEIGHT, _WIDTH, 3]
    )
    model = build_onnx(
        tmp_path / "identity.onnx",
        [("input", shape, TensorProto.FLOAT)],
        [("output", shape, TensorProto.FLOAT)],
    )
    # Keep a real operation so OpenVINO does not collapse an Identity's input
    # and output names into an alias. All normalized test values are positive.
    graph = onnx.load(model)
    graph.graph.node[0].op_type = "Relu"
    onnx.save(graph, model)
    config_path = write_config(
        tmp_path,
        "normalization",
        {
            "input_model": str(model),
            "inputs": [
                {
                    "name": "input",
                    "layout": layout,
                    "encoding": {"from": encoding, "to": "BGR"},
                    "mean_values": [10, 20, 30],
                    "scale_values": [2, 4, 8],
                }
            ],
            "onnx_simplification": False,
            "onnx_optimizations": False,
        },
    )
    options = platform_options(Platform.RVC2)
    output_name = f"_channel-normalization-{layout}-{encoding}"
    convert(
        Platform.RVC2,
        *options,
        path=str(config_path),
        output_dir=output_name,
        to="native",
    )
    cfg, _, _ = get_configs(Platform.RVC2, str(config_path), list(options))
    stage = next(iter(cfg.stages.values()))
    inferer = get_inferer(
        Platform.RVC2,
        str(locate_converted_model(OUTPUTS_DIR / output_name, "rvc2")),
        color_image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    actual = inferer.infer({"input": color_image})["output"]
    pixels = reorder_layout(
        np.asarray(actual, dtype=np.float32), layout, "NHWC"
    )
    image = cv2.imread(str(color_image))
    assert image is not None
    channels = image.astype(np.float32)
    if encoding == "RGB":
        channels = channels[..., ::-1]
    expected = (channels[None] - [10, 20, 30]) / [2, 4, 8]
    np.testing.assert_allclose(
        pixels, np.broadcast_to(expected, pixels.shape), atol=0.05
    )
