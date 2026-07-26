"""Conversion tests for less-common input / calibration variants.

Each converts a tiny toy model on one platform, covering a path the main
conversion tests miss:

  * **RVC4 with ``.raw`` calibration** -- SNPE consumes ``.raw`` calibration
    files verbatim (the ``e.path.suffix == ".raw"`` branch in
    ``rvc4/exporter.prepare_calibration_data``), bypassing the image reader
    every other test exercises.
  * **RVC2 with a 3D ``HWC`` input** -- a rank-3 channels-last input rather
    than the usual 4D ``NCHW``.
  * **Hailo with calibration disabled** -- no calibration data at all; the
    ``_disable_calibration`` branch returns the float ``.har`` directly.

(The toy ``.tflite`` conversion + precision lives in ``test_toy_tflite``.)

Run inside the platform Docker image, e.g.::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k raw_calibration'
"""

from pathlib import Path

import pytest
import yaml

from modelconverter.__main__ import convert
from modelconverter.utils.types import Target
from tests.helpers.conversion import assert_produced, write_toy_conv_config
from tests.helpers.onnx_factory import build_hwc_onnx


@pytest.mark.rvc4
def test_rvc4_raw_calibration(tmp_path: Path):
    config = write_toy_conv_config(tmp_path, calibration="raw")
    output_name = "_rvc4-raw-calib"
    convert(Target.RVC4, path=str(config), output_dir=output_name, to="native")
    assert_produced(output_name)


@pytest.mark.hailo
def test_hailo_without_calibration(tmp_path: Path):
    # No calibration data; `disable_calibration` returns the float HAR.
    config = write_toy_conv_config(tmp_path, calibration="none")
    output_name = "_hailo-no-calib"
    convert(
        Target.HAILO,
        "hailo.disable_calibration",
        "True",
        path=str(config),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)


@pytest.mark.rvc2
def test_rvc2_hwc_input(tmp_path: Path):
    onnx_path = build_hwc_onnx(tmp_path / "hwc.onnx", size=32)
    config = {
        "input_model": str(onnx_path),
        "inputs": [
            {
                "name": "data",
                "shape": [32, 32, 3],
                "layout": "HWC",
                "data_type": "float32",
                # Raw input: no baked preprocessing, so the model optimizer
                # need not disambiguate the channel axis of this 3D input.
                "encoding": "NONE",
            }
        ],
        "outputs": [{"name": "out"}],
    }
    config_path = tmp_path / "hwc.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    output_name = "_rvc2-hwc"
    convert(
        Target.RVC2, path=str(config_path), output_dir=output_name, to="native"
    )
    assert_produced(output_name)
