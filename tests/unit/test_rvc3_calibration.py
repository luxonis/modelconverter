"""RVC3 calibration tests that do not require the OpenVINO toolchain."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modelconverter.cli.utils import extract_preprocessing
from modelconverter.platforms.rvc3 import exporter as rvc3_exporter
from modelconverter.platforms.rvc3.exporter import RVC3Exporter
from modelconverter.utils.config import Config
from tests.helpers.onnx_factory import single_io_onnx


def test_externalized_preprocessing_uses_float_numpy_pot_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = single_io_onnx(
        tmp_path / "model.onnx",
        shape=[1, 3, 1, 1],
        output_shape=[1, 3, 1, 1],
    ).resolve()
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    Image.fromarray(np.array([[[100, 110, 120]]], dtype=np.uint8)).save(
        calibration_dir / "pixel.png"
    )
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": [1, 3, 1, 1],
            "layout": "NCHW",
            "encoding": {"from": "RGB", "to": "BGR"},
            "mean_values": [10, 20, 30],
            "scale_values": 2,
            "calibration": {"path": str(calibration_dir)},
            "onnx_simplification": False,
        },
    )
    extract_preprocessing(config)
    output_dir = tmp_path / "rvc3-output"
    output_dir.mkdir(exist_ok=True)
    exporter = RVC3Exporter(next(iter(config.stages.values())), output_dir)
    commands: list[list[object]] = []
    monkeypatch.setattr(
        rvc3_exporter,
        "subprocess_run",
        lambda command, **_kwargs: commands.append(command),
    )
    xml_path = tmp_path / "model.xml"

    exporter._calibrate(xml_path)

    pot_config = json.loads(
        (exporter.intermediate_outputs_dir / "pot_config.json").read_text()
    )
    dataset = pot_config["engine"]["datasets"][0]
    assert dataset["reader"] == "numpy_reader"
    tensor_path = Path(dataset["data_source"]) / "0.npy"
    actual = np.load(tensor_path)
    assert actual.dtype == np.float32
    assert actual.shape == (1, 3, 1, 1)
    np.testing.assert_array_equal(
        actual,
        np.array([[[[45.0]], [[45.0]], [[45.0]]]], dtype=np.float32),
    )
    assert commands
    assert commands[0][0] == "pot"
