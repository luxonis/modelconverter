"""Conversion of an ONNX model whose weights live in an external-data file.

Large ONNX models store their weights in a sibling ``<name>_data`` file
rather than embedding them in the ``.onnx`` graph (what
``onnx.save(save_as_external_data=True)`` / modelconverter's own
``save_onnx_model`` produce). modelconverter has to notice that sibling and
carry it alongside the model through the whole pipeline:

  * ``base_exporter.__init__`` copies the ``<name>_data`` file next to the
    sanitized model in both ``intermediate_outputs/`` and the output dir
    (``base_exporter.py`` lines 84-92);
  * ``simplify_onnx`` / ``onnx_tools`` / ``config.generate_renamed_onnx``
    re-save the model as external data so the sibling is preserved across
    every rewrite.

None of that is otherwise exercised (no other test builds an external-data
model). Here we build the toy conv net with its weights externalized and
convert it on every backend, asserting success -- the model is only usable
if the sibling weights travelled with it.

Runs inside the platform Docker image, e.g.::

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

ALL_PLATFORMS = ("rvc2", "rvc3", "rvc4", "hailo")


@pytest.fixture(scope="module")
def external_data_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the toy conv ONNX with external weights + calibration + config.

    The ``.onnx`` here holds no weights on its own -- they are in the sibling
    ``toy_conv.onnx_data`` -- so a conversion only succeeds if modelconverter
    keeps the two together.
    """
    workdir = tmp_path_factory.mktemp("external_data")
    config_path = write_toy_conv_config(workdir, external_data=True)
    assert (workdir / "toy_conv.onnx_data").exists(), (
        "factory did not externalize the weights"
    )
    return config_path


@pytest.mark.parametrize(
    "platform",
    [
        pytest.param(p, marks=getattr(pytest.mark, p), id=p)
        for p in ALL_PLATFORMS
    ],
)
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
