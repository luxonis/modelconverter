"""Conversion of an ONNX model whose weights live in an external-data file.

Large ONNX models keep their weights in a sibling ``<name>_data`` file rather
than embedding them in the graph, and modelconverter has to carry that sibling
alongside the model through the whole pipeline: ``base_exporter`` copies it next
to the sanitized model in both ``intermediate_outputs/`` and the output dir, and
``simplify_onnx`` / ``onnx_tools`` / ``generate_renamed_onnx`` re-save as external
data so it survives every rewrite.

No other test builds an external-data model, so this is the only thing exercising
that. Converting on every backend and asserting success is enough: the model is
only usable at all if the sibling weights travelled with it.

Run inside the platform Docker image, e.g.::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k external_data'
"""

from pathlib import Path

import pytest

from modelconverter.__main__ import convert
from modelconverter.utils.types import Target
from tests.helpers.conversion import (
    HAILO_FAST_OPTS,
    assert_produced,
    write_toy_conv_config,
)
from tests.helpers.platforms import platform_params


@pytest.fixture(scope="module")
def external_data_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("external_data")
    config_path = write_toy_conv_config(workdir, external_data=True)
    assert (workdir / "toy_conv.onnx_data").exists(), (
        "factory did not externalize the weights"
    )
    return config_path


@pytest.mark.parametrize("platform", platform_params())
def test_external_data(platform: str, external_data_config: Path):
    output_name = f"_external-data-{platform}"
    extra = HAILO_FAST_OPTS if platform == "hailo" else ()
    convert(
        Target(platform),
        *extra,
        path=str(external_data_config),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)
