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

import pytest

from modelconverter.__main__ import convert
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target
from tests.helpers.conversion import write_toy_conv_config


@pytest.fixture(scope="module")
def compile_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Toy conv ONNX + calibration dir + config (calibration is required:
    compilation runs *after* quantization, so a disabled-calibration run would
    return the float HAR and never reach the compile step)."""
    return write_toy_conv_config(tmp_path_factory.mktemp("hailo_compile"))


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
