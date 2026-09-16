"""Host-side tests for Hailo calibration tensor preparation."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from modelconverter.cli.utils import extract_preprocessing
from modelconverter.utils.config import Config, ImageCalibrationConfig
from tests.helpers.onnx_factory import single_io_onnx


class _Runner:
    def __init__(self, input_shape: list[int] | None = None) -> None:
        self.input_shape = input_shape or [1, 1, 1, 3]

    def get_hn_dict(self) -> dict[str, Any]:
        return {
            "layers": {
                "hailo_input": {
                    "original_names": ["input0"],
                    "input_shapes": [self.input_shape],
                }
            }
        }


def test_externalized_preprocessing_reaches_hailo_in_model_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hailo receives HWC float data after archive-side preprocessing."""
    hailo_sdk = ModuleType("hailo_sdk_client")
    hailo_sdk.ClientRunner = object  # type: ignore[attr-defined]
    hailo_sdk.__version__ = "test"  # type: ignore[attr-defined]
    tensorflow = ModuleType("tensorflow")
    tensorflow.config = SimpleNamespace(  # type: ignore[attr-defined]
        list_physical_devices=lambda _kind: []
    )
    monkeypatch.setitem(sys.modules, "hailo_sdk_client", hailo_sdk)
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    module_name = "modelconverter.platforms.hailo.exporter"
    sys.modules.pop(module_name, None)

    try:
        hailo_exporter = importlib.import_module(module_name)
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
        stage = next(iter(config.stages.values()))
        exporter = object.__new__(hailo_exporter.HailoExporter)
        exporter._inputs = {inp.name: inp for inp in stage.inputs}

        data = exporter._get_calibration_data(_Runner())

        actual = data["hailo_input"]
        assert actual.dtype == np.float32
        assert actual.shape == (1, 1, 1, 3)
        np.testing.assert_array_equal(
            actual,
            np.array([[[[45.0, 45.0, 45.0]]]], dtype=np.float32),
        )

        # A user tensor is already backend-ready HWC data. It must remain
        # opaque even though the source model and retained snapshot are NCHW
        # and contain non-identity preprocessing.
        (calibration_dir / "pixel.png").unlink()
        source = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        np.save(calibration_dir / "model-domain.npy", source)
        stage.inputs[0].calibration = ImageCalibrationConfig(
            path=calibration_dir
        )

        tensor_data = exporter._get_calibration_data(_Runner())

        np.testing.assert_array_equal(tensor_data["hailo_input"][0], source)

        # Shape equality must not hide the NCHW -> HWC conversion for managed
        # tensors when all three non-batch dimensions happen to be equal.
        cubic_model = single_io_onnx(
            tmp_path / "cubic.onnx",
            shape=[1, 3, 3, 3],
            output_shape=[1, 3, 3, 3],
        ).resolve()
        cubic_dir = tmp_path / "cubic-calibration"
        cubic_dir.mkdir()
        cubic_source = np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3)
        np.save(cubic_dir / "sample.npy", cubic_source)
        cubic_config = Config.get_config(
            None,
            {
                "input_model": str(cubic_model),
                "shape": [1, 3, 3, 3],
                "layout": "NCHW",
                "encoding": "RGB",
                "mean_values": [1, 2, 3],
                "calibration": {"path": str(cubic_dir)},
                "onnx_simplification": False,
            },
        )
        extract_preprocessing(cubic_config)
        cubic_input = next(iter(cubic_config.stages.values())).inputs[0]
        cubic_calibration = cubic_input.calibration
        assert isinstance(cubic_calibration, ImageCalibrationConfig)
        cubic_calibration._generated_from_random = True
        exporter._inputs = {cubic_input.name: cubic_input}

        cubic_data = exporter._get_calibration_data(_Runner([1, 3, 3, 3]))[
            "hailo_input"
        ][0]

        means = np.array([1, 2, 3], dtype=np.float32).reshape(1, 3, 1, 1)
        expected = (cubic_source - means).transpose(0, 2, 3, 1)[0]
        np.testing.assert_array_equal(cubic_data, expected)
    finally:
        sys.modules.pop(module_name, None)
