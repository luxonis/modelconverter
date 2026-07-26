"""Toy TFLite conversion + precision across **all platforms**.

A hand-built single-op ``.tflite`` (see ``tests/helpers/tflite_factory``) is
converted on every platform, exercising each backend's distinct TFLite entry
path -- ``tflite2onnx`` (rvc2/rvc3), ``snpe-tflite-to-dlc`` (rvc4) and
``translate_tf_model`` (hailo). The model is tiny enough that hailo does a
full HEF compile.

**Precision**: the model has no weights and no baked math beyond a ``Relu``, so
a raw image round-trips to a (channel-permuted, quantized) copy of itself.
Comparing the *sorted* output values to the sorted reference is robust to each
backend's output layout / channel order while still catching quantization
error -- so one comparison works for every platform. A TFLite graph is NHWC,
which the vendor inferers now feed correctly (see ``read_image``'s ``layout``).

Runs inside the platform Docker image, e.g.::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k toy_tflite'
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.packages.getters import get_inferer
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target
from tests.helpers.conversion import assert_produced
from tests.helpers.precision import cosine_similarity, locate_converted_model
from tests.helpers.tflite_factory import build_toy_tflite

ALL_PLATFORMS = ("rvc2", "rvc3", "rvc4", "hailo")
_THRESHOLD = 0.9
# Three distinct, positive channel values inside the default random-calibration
# range (mean 127.5, std 35). Positive so the Relu is a clean identity, and
# well-separated so the quantizers keep them apart.
_CHANNEL_VALUES = (90, 130, 170)

_PARAMS = [
    pytest.param(p, marks=getattr(pytest.mark, p), id=p) for p in ALL_PLATFORMS
]


@pytest.fixture(scope="module")
def toy_tflite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("toy_tflite")
    return build_toy_tflite(workdir / "toy.tflite")


@pytest.fixture(scope="module")
def solid_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An 8x8 image with a distinct constant value per channel."""
    path = tmp_path_factory.mktemp("toy_tflite_img") / "solid.png"
    img = np.full((8, 8, 3), _CHANNEL_VALUES, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


@pytest.mark.parametrize("platform", _PARAMS)
def test_toy_tflite_conversion(platform: str, toy_tflite: Path):
    output_name = f"_toy-tflite-{platform}"
    convert(
        Target(platform),
        path=str(toy_tflite),
        output_dir=output_name,
        to="native",
    )
    if platform == "hailo":
        # Tiny enough for the real HEF compile (no `disable_compilation`).
        hefs = list((OUTPUTS_DIR / output_name).rglob("*.hef"))
        assert hefs, f"no compiled .hef produced for {output_name}"
    else:
        assert_produced(output_name)


@pytest.mark.parametrize("platform", _PARAMS)
def test_toy_tflite_precision(
    platform: str, toy_tflite: Path, solid_image: Path
):
    target = Target(platform)
    output_name = f"_toy-tflite-prec-{platform}"
    # Hailo precision runs SDK_QUANTIZED emulation on the quantized HAR, so
    # skip the slow HEF compile here (fidelity is unaffected).
    opts = ["hailo.disable_compilation", "True"] if platform == "hailo" else []
    convert(
        target,
        *opts,
        path=str(toy_tflite),
        output_dir=output_name,
        to="native",
    )

    # A bare model path is not a config file -- pass it as an `input_model`
    # override with `path=None`, exactly as `convert` does internally.
    cfg, _, _ = get_configs(
        target, None, ["input_model", str(toy_tflite), *opts]
    )
    stage = next(iter(cfg.stages.values()))
    model_path = locate_converted_model(OUTPUTS_DIR / output_name, platform)
    inferer = get_inferer(
        target,
        str(model_path),
        solid_image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer({stage.inputs[0].name: solid_image})

    (output,) = converted.values()
    values = np.sort(np.asarray(output, dtype=np.float32).ravel())
    per_channel = values.size // len(_CHANNEL_VALUES)
    reference = np.sort(
        np.repeat(np.array(_CHANNEL_VALUES, dtype=np.float32), per_channel)
    )
    cos = cosine_similarity(reference, values)
    assert cos >= _THRESHOLD, (
        f"{platform} tflite precision: cosine {cos:.5f} < {_THRESHOLD}"
    )
