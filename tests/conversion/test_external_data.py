"""Conversion of an ONNX model whose weights live in an external-data file.

Large ONNX models keep their weights in a sibling ``<name>_data`` file rather
than embedding them in the graph, and modelconverter has to carry that sibling
alongside the model through the whole pipeline: ``base_exporter`` copies it next
to the sanitized model in both ``intermediate_outputs/`` and the output dir, and
``_simplify_onnx`` / ``onnx_tools`` / ``generate_renamed_onnx`` re-save as external
data so it survives every rewrite.

This is the only thing exercising that end to end; the host-side tests cover the
bookkeeping alone (which files staging copies, where they are looked up).
Converting on every backend and asserting success is enough: the model is only
usable at all if the sibling weights travelled with it.

Run inside the platform Docker image, e.g.::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k external_data'
"""

from pathlib import Path

import pytest

from modelconverter.__main__ import convert
from modelconverter.utils.types import Platform
from tests.helpers.conversion import (
    HAILO_FAST_OPTS,
    assert_produced,
    write_toy_conv_config,
)
from tests.helpers.platform_options import platform_options
from tests.helpers.platforms import platform_params


@pytest.fixture(scope="module")
def external_data_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("external_data")
    config_path = write_toy_conv_config(workdir, external_data=True)
    assert (workdir / "toy_conv.onnx_data").exists(), (
        "factory did not externalize the weights"
    )
    return config_path


@pytest.mark.parametrize("platform_name", platform_params())
def test_external_data(platform_name: str, external_data_config: Path):
    platform = Platform(platform_name)
    output_name = f"_external-data-{platform_name}"
    extra = HAILO_FAST_OPTS if platform_name == "hailo" else ()
    convert(
        platform,
        *platform_options(platform),
        *extra,
        path=str(external_data_config),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)
