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

ALL_PLATFORMS = ("rvc2", "rvc3", "rvc4", "hailo")

# Hailo: skip the (slow) HEF compile -- a successful quantized conversion is
# enough to prove the external weights travelled with the model.
_HAILO_FAST_OPTS = (
    "hailo.compression_level",
    "0",
    "hailo.optimization_level",
    "0",
    "hailo.disable_compilation",
    "True",
)


@pytest.fixture(scope="module")
def external_data_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the toy conv ONNX with external weights + calibration + config.

    The ``.onnx`` here holds no weights on its own -- they are in the sibling
    ``toy_conv.onnx_data`` -- so a conversion only succeeds if modelconverter
    keeps the two together.
    """
    workdir = tmp_path_factory.mktemp("external_data")
    onnx_path = workdir / "toy_conv.onnx"
    build_toy_conv_onnx(onnx_path, size=_SIZE, external_data=True)
    assert (workdir / "toy_conv.onnx_data").exists(), (
        "factory did not externalize the weights"
    )

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


@pytest.mark.parametrize(
    "platform",
    [
        pytest.param(p, marks=getattr(pytest.mark, p), id=p)
        for p in ALL_PLATFORMS
    ],
)
def test_external_data(platform: str, external_data_config: Path):
    output_name = f"_external-data-{platform}"
    extra = _HAILO_FAST_OPTS if platform == "hailo" else ()
    convert(
        Target(platform),
        *extra,
        path=str(external_data_config),
        output_dir=output_name,
        to="native",
    )
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    assert any(out_dir.iterdir()), f"no output produced in {out_dir}"
