"""Toy conv-net numeric-fidelity (precision) tests -- **all platforms**.

``build_toy_integration_onnx`` is great for exercising the preprocessing
surface, but it is a poor fidelity subject: its output is essentially the
preprocessed input, and its deliberately wide per-input scale spread makes
the Hailo fixed-point quantizer reject it ("shift delta"). So the toy
integration precision check only covers rvc2/rvc4.

This module uses a different, well-conditioned toy -- a single-input 3x3
``Conv`` (``build_toy_conv_onnx``) with modest weights -- which **every**
backend, Hailo included, can quantize and reproduce near-losslessly. It
still carries mean/scale + channel reversal in the config, so the converted
model exercises the full baked-preprocessing path; the difference is only
that the graph now does real (quantizable) work.

For each platform we:

  1. convert the toy conv net (with a directory of varied calibration
     images, so the quantization ranges are healthy);
  2. build an fp32 golden by baking the same preprocessing into the ONNX
     (``golden_reference_outputs``) and running it on a constant image;
  3. run the converted model's vendor inferer on that same constant image;
  4. assert the outputs match by cosine similarity.

Runs inside the platform Docker image, e.g.::

    modelconverter shell hailo --dev --no-gpu -c 'python -m pytest -k toy_precision'
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
from tests.helpers.conversion import write_toy_conv_config
from tests.helpers.precision import (
    cosine_similarity,
    golden_reference_outputs,
    locate_converted_model,
)

_SIZE = 32
# Constant value the fidelity comparison runs on (within the calibration
# range below, so the quantized model represents it well).
_INFER_VALUE = 100
_THRESHOLD = 0.9

ALL_PLATFORMS = ("rvc2", "rvc3", "rvc4", "hailo")

# Hailo: skip the slow HEF compile -- the precision check runs
# `SDK_QUANTIZED` inference on the quantized HAR, so compilation is unneeded.
_PLATFORM_OPTS: dict[str, tuple[str, ...]] = {
    "hailo": ("hailo.disable_compilation", "True"),
}


@pytest.fixture(scope="module")
def toy_conv_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the toy conv ONNX + varied calibration dir + config."""
    return write_toy_conv_config(
        tmp_path_factory.mktemp("toy_precision"), size=_SIZE
    )


@pytest.fixture(scope="module")
def constant_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("toy_prec_const") / "const.png"
    cv2.imwrite(
        str(path), np.full((_SIZE, _SIZE, 3), _INFER_VALUE, dtype=np.uint8)
    )
    return path


def _to_nchw(arr: np.ndarray, platform: str) -> np.ndarray:
    """Normalize a vendor inferer's output to NCHW for comparison.

    The Hailo inferer returns spatial outputs channel-last (HWC / NHWC); the
    others already return NCHW. The golden reference is NCHW.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and platform == "hailo":
        return arr.transpose(2, 0, 1)[np.newaxis]
    return arr


@pytest.mark.parametrize(
    "platform",
    [
        pytest.param(p, marks=getattr(pytest.mark, p), id=p)
        for p in ALL_PLATFORMS
    ],
)
def test_toy_precision(
    platform: str, toy_conv_config: Path, constant_image: Path
):
    target = Target(platform)
    opts = _PLATFORM_OPTS.get(platform, ())
    output_name = f"_toy-prec-{platform}"

    convert(
        target,
        *opts,
        path=str(toy_conv_config),
        output_dir=output_name,
        to="native",
    )

    cfg, _, _ = get_configs(target, str(toy_conv_config), list(opts))
    stage = next(iter(cfg.stages.values()))

    golden_dir = OUTPUTS_DIR / f"{output_name}_golden"
    reference = golden_reference_outputs(
        Path(stage.input_model),
        {inp.name: inp for inp in stage.inputs},
        golden_dir,
        float(_INFER_VALUE),
    )

    model_path = locate_converted_model(OUTPUTS_DIR / output_name, platform)
    inferer = get_inferer(
        target,
        str(model_path),
        constant_image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer({"img": constant_image})

    for (ref_name, ref), conv in zip(
        reference.items(), converted.values(), strict=True
    ):
        conv = _to_nchw(conv, platform)
        cos = cosine_similarity(ref, conv)
        assert cos >= _THRESHOLD, (
            f"{platform} output {ref_name!r}: cosine {cos:.5f} < {_THRESHOLD}"
        )
