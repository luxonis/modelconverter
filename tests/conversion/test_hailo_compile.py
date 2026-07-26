"""Hailo full-compilation test.

Every other Hailo test sets ``hailo.disable_compilation = True`` -- that
stops after quantization and returns the quantized ``.har`` (it is enough
for a conversion smoke check or a fidelity check, which runs
``SDK_QUANTIZED`` inference on the HAR). It never exercises the actual HEF
compile: ``ClientRunner.compile()`` and the ``.hef`` write
(``hailo/exporter.py`` lines 96-101).

This test runs the *real* compile on the well-conditioned toy conv net
(``build_toy_conv_onnx`` -- the same small model the Hailo precision test
uses, so it quantizes cleanly) and asserts a ``.hef`` is produced. On a
GPU-less CI host the Hailo SDK forces optimization/compression to 0, so the
compile of this tiny model stays cheap.

Runs inside the Hailo Docker image::

    modelconverter shell hailo --dev --no-gpu -c 'python -m pytest -k hailo_compile'
"""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def compile_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Toy conv ONNX + varied calibration dir + config (calibration is
    required: compilation runs *after* quantization, so a disabled-calibration
    run would return the float HAR and never reach the compile step)."""
    workdir = tmp_path_factory.mktemp("hailo_compile")
    onnx_path = workdir / "toy_conv.onnx"
    build_toy_conv_onnx(onnx_path, size=_SIZE)

    calib_dir = workdir / "calib"
    calib_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(16):
        cv2.imwrite(
            str(calib_dir / f"{i}.png"),
            rng.integers(0, 256, (_SIZE, _SIZE, 3), dtype=np.uint8),
        )

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
    return config_path


@pytest.mark.hailo
def test_hailo_compile(compile_config: Path):
    output_name = "_hailo-compile"
    # Note: NO `disable_compilation` -- we want the real HEF compile. Keep the
    # optimization/compression cheap (the GPU-less host forces them to 0 too).
    convert(
        Target.HAILO,
        "hailo.optimization_level",
        "0",
        "hailo.compression_level",
        "0",
        path=str(compile_config),
        output_dir=output_name,
        to="native",
    )

    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    hefs = list(out_dir.rglob("*.hef"))
    assert hefs, f"no compiled .hef produced in {out_dir}"
    assert hefs[0].stat().st_size > 0, f"compiled HEF {hefs[0]} is empty"
