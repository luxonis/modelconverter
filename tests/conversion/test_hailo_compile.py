"""Hailo full-compilation test.

Every other Hailo test sets ``hailo.disable_compilation``, which stops after
quantization and returns the quantized ``.har`` -- enough for a conversion smoke
check or a fidelity check, but it never reaches ``ClientRunner.compile()`` or the
``.hef`` write. This runs the real compile.

The subject is the well-conditioned toy conv net (the same one the Hailo precision
test uses, so it quantizes cleanly). Calibration is required: compilation runs
*after* quantization, so a disabled-calibration run would return the float HAR and
never reach the compile step. On a GPU-less CI host the SDK forces
optimization/compression to 0 anyway, so compiling this tiny model stays cheap.

Run inside the Hailo Docker image::

    modelconverter shell hailo --dev --no-gpu -c 'python -m pytest -k hailo_compile'
"""

from pathlib import Path

import pytest

from modelconverter.__main__ import convert
from modelconverter.utils.types import Platform
from tests.helpers.conversion import (
    assert_produced_suffix,
    write_toy_conv_config,
)


@pytest.fixture(scope="module")
def compile_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_toy_conv_config(tmp_path_factory.mktemp("hailo_compile"))


@pytest.mark.hailo
def test_hailo_compile(compile_config: Path):
    output_name = "_hailo-compile"
    # No `disable_compilation` here -- we want the real HEF compile.
    convert(
        Platform.HAILO,
        "hailo.optimization_level",
        "0",
        "hailo.compression_level",
        "0",
        path=str(compile_config),
        output_dir=output_name,
        to="native",
    )
    assert_produced_suffix(output_name, ".hef")
