"""RVC4 conversion driven by a custom ``encodings.json``.

The unit tests cover *parsing* a custom ``encodings.json`` (``rvc4.encodings``
accepting an inline dict / JSON string / path, the per-channel expansion,
the ``--quantization_overrides`` extraction). What they cannot cover, being
host-only, is the RVC4 **exporter** actually consuming those encodings and
the two related SNPE knobs -- so this e2e fills that gap. It converts the toy
conv net with:

  * a custom ``encodings.json`` (a per-channel ``param`` encoding for the
    conv weight + an ``activation`` encoding) referenced by path, which drives
    ``RVC4Exporter.generate_io_encodings`` -> ``snpe-onnx-to-dlc
    --quantization_overrides`` and ``snpe-dlc-quant --override_params``
    (``rvc4/exporter.py`` lines 180-181, 267-279, 335-343);
  * ``use_per_row_quantization`` -> ``snpe-dlc-quant
    --use_per_row_quantization`` (default off, so otherwise never exercised);
  * a non-default ``htp_socs`` -> ``snpe-dlc-graph-prepare --htp_socs``.

It asserts a quantized ``.dlc`` is produced. ``generate_io_encodings``
normalizes the *exposed* input/output activation encodings to default int8
IO regardless of what we pass, so the custom values mainly exercise the
internal-tensor / param path; the point here is the exporter code, not
numeric fidelity.

Runs inside the RVC4 Docker image::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k rvc4_encodings'
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from modelconverter.__main__ import convert
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target
from tests.helpers.onnx_factory import build_toy_conv_onnx

_SIZE = 32
_OUT_CHANNELS = 4

# A custom AIMET-style encodings file. The `weight` param uses per-channel
# vectors (one entry per output channel) so parsing also exercises the
# per-channel expansion; `img`/`out` activations are normalized to int8 IO
# by the exporter but still drive the --quantization_overrides path.
_ENCODINGS = {
    "activation_encodings": {
        "img": [
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": "False",
                "min": 0.0,
                "max": 255.0,
                "scale": 1.0,
                "offset": 0,
            }
        ],
        "out": [
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": "False",
                "min": 0.0,
                "max": 255.0,
                "scale": 1.0,
                "offset": 0,
            }
        ],
    },
    "param_encodings": {
        "weight": [
            {
                "bitwidth": 8,
                "dtype": "int",
                "is_symmetric": "True",
                "min": [-0.5] * _OUT_CHANNELS,
                "max": [0.5] * _OUT_CHANNELS,
                "scale": [0.004] * _OUT_CHANNELS,
                "offset": [-128] * _OUT_CHANNELS,
            }
        ],
    },
}


@pytest.fixture(scope="module")
def encodings_config(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    """Toy conv ONNX + calibration + config, and a sibling ``encodings.json``.

    Returns ``(config_path, encodings_path)``; the encodings file is passed to
    the conversion via the ``rvc4.encodings`` override (a path).
    """
    workdir = tmp_path_factory.mktemp("rvc4_encodings")
    onnx_path = workdir / "toy_conv.onnx"
    build_toy_conv_onnx(onnx_path, size=_SIZE, out_channels=_OUT_CHANNELS)

    calib_dir = workdir / "calib"
    calib_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(16):
        cv2.imwrite(
            str(calib_dir / f"{i}.png"),
            rng.integers(0, 256, (_SIZE, _SIZE, 3), dtype=np.uint8),
        )

    encodings_path = workdir / "encodings.json"
    encodings_path.write_text(json.dumps(_ENCODINGS, indent=2))

    config = {
        "input_model": str(onnx_path),
        "inputs": [
            {
                "name": "img",
                "shape": [1, 3, _SIZE, _SIZE],
                "data_type": "float32",
                "mean_values": [128, 128, 128],
                "scale_values": [58, 58, 58],
                "encoding": {"from": "RGB", "to": "BGR"},
                "calibration": {"path": str(calib_dir), "max_images": 16},
            }
        ],
        "outputs": [{"name": "out"}],
    }
    config_path = workdir / "toy_conv.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, encodings_path


@pytest.mark.rvc4
def test_rvc4_encodings(encodings_config: tuple[Path, Path]):
    config_path, encodings_path = encodings_config
    output_name = "_rvc4-encodings"
    convert(
        Target.RVC4,
        # Custom encodings.json (by path) + per-row quantization + a
        # non-default HTP SoC. `ast.literal_eval` parses the list/bool opts;
        # the path string falls through as-is for `rvc4.encodings`.
        "rvc4.encodings",
        str(encodings_path),
        "rvc4.use_per_row_quantization",
        "True",
        "rvc4.use_per_channel_quantization",
        "False",
        "rvc4.htp_socs",
        "['sm8650']",
        path=str(config_path),
        output_dir=output_name,
        to="native",
    )

    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    dlcs = list(out_dir.glob("*.dlc")) or list(out_dir.rglob("*.dlc"))
    assert dlcs, f"no converted .dlc produced in {out_dir}"
