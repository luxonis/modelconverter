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
import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.platforms.getters import get_inferer
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Platform
from tests.helpers.platform_options import platform_options
from tests.helpers.platforms import platform_params
from tests.helpers.precision import RecordProperty, locate_converted_model
from tests.helpers.tflite_factory import build_toy_tflite

PLATFORMS = ("rvc2", "rvc3", "rvc4")
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
    return build_toy_tflite(workdir / "toy.tflite")


@pytest.fixture(scope="module")
def solid_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create an 8x8 image with a distinct constant value per channel."""
    path = tmp_path_factory.mktemp("toy_tflite_img") / "solid.png"
    cv2.imwrite(str(path), np.full((8, 8, 3), _CHANNEL_VALUES, dtype=np.uint8))
    return path


@pytest.mark.parametrize("platform_name", _PARAMS)
def test_toy_tflite_precision(
    platform_name: str,
    toy_tflite: Path,
    solid_image: Path,
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
        solid_image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer({stage.inputs[0].name: solid_image})

    (output,) = converted.values()
    output_array = np.asarray(output, dtype=np.float32)
    # The image is 8x8, so the channel axis is the only axis of size 3,
    # whatever the output layout of the backend.
    pixels = np.moveaxis(output_array, output_array.shape.index(3), -1)
    # OpenCV wrote the image as BGR, and the model gets it as RGB.
    expected_rgb = np.asarray(_CHANNEL_VALUES[::-1], dtype=np.float32)
    max_error = float(np.abs(pixels - expected_rgb).max())
    record_property(f"{platform_name} max error", round(max_error, 3))
    assert max_error <= _TOLERANCE, (
        f"{platform_name} tflite output differs from the RGB input by "
        f"{max_error:.3f}: expected {expected_rgb.tolist()}, got channel "
        f"means {pixels.reshape(-1, 3).mean(axis=0).tolist()}"
    )
