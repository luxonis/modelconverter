"""Host-side tests for Hailo calibration tensor preparation."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol

import numpy as np
import pytest
from PIL import Image

from modelconverter.cli.utils import extract_preprocessing
from modelconverter.utils import (
    ModelconverterException,
    PreprocessingEmbeddingError,
)
from modelconverter.utils.config import (
    Config,
    ImageCalibrationConfig,
    InputConfig,
)
from tests.helpers.onnx_factory import single_io_onnx

# The HN layers, as `_get_hn_layer_info` reads them out of the Hailo IR.
_HnLayers = dict[str, dict[str, list[str] | list[list[int]]]]


class _Runner:
    def __init__(self, input_shape: list[int] | None = None) -> None:
        self.input_shape = input_shape or [1, 1, 1, 3]

    def get_hn_dict(self) -> dict[str, _HnLayers]:
        return {
            "layers": {
                "hailo_input": {
                    "original_names": ["input0"],
                    "input_shapes": [self.input_shape],
                }
            }
        }


class _CalibrationExporter(Protocol):
    """The slice of `HailoExporter` these tests drive."""

    def _get_calibration_data(
        self, runner: _Runner
    ) -> dict[str, np.ndarray]: ...


class _FakeHailoSdk(ModuleType):
    ClientRunner = object
    __version__ = "test"


class _FakeTensorflow(ModuleType):
    config = SimpleNamespace(list_physical_devices=lambda _kind: [])


@pytest.fixture
def hailo_exporter_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        sys.modules, "hailo_sdk_client", _FakeHailoSdk("hailo_sdk_client")
    )
    monkeypatch.setitem(
        sys.modules, "tensorflow", _FakeTensorflow("tensorflow")
    )
    module_name = "modelconverter.platforms.hailo.exporter"
    sys.modules.pop(module_name, None)

    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)


def _externalized_exporter(
    hailo_exporter_module: ModuleType,
    tmp_path: Path,
    calibration_dir: Path,
    *,
    shape: list[int] | None = None,
    encoding: str | dict[str, str] = "RGB",
    mean_values: list[int] | int = 0,
) -> tuple[_CalibrationExporter, InputConfig]:
    shape = shape or [1, 3, 1, 1]
    model = single_io_onnx(
        tmp_path / "model.onnx",
        shape=shape,
        output_shape=shape,
    ).resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": shape,
            "layout": "NCHW",
            "encoding": encoding,
            "mean_values": mean_values,
            "scale_values": 2,
            "calibration": {"path": str(calibration_dir)},
            "onnx_simplification": False,
        },
    )
    extract_preprocessing(config)
    stage = next(iter(config.stages.values()))
    exporter = object.__new__(hailo_exporter_module.HailoExporter)
    exporter._inputs = {inp.name: inp for inp in stage.inputs}
    return exporter, stage.inputs[0]


def test_externalized_preprocessing_reaches_hailo_in_model_domain(
    hailo_exporter_module: ModuleType, tmp_path: Path
) -> None:
    """Hailo receives HWC float data after archive-side preprocessing."""
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    Image.fromarray(np.array([[[100, 110, 120]]], dtype=np.uint8)).save(
        calibration_dir / "pixel.png"
    )
    exporter, _ = _externalized_exporter(
        hailo_exporter_module,
        tmp_path,
        calibration_dir,
        encoding={"from": "RGB", "to": "BGR"},
        mean_values=[10, 20, 30],
    )

    actual = exporter._get_calibration_data(_Runner())["hailo_input"]

    assert actual.dtype == np.float32
    assert actual.shape == (1, 1, 1, 3)
    np.testing.assert_array_equal(
        actual,
        np.array([[[[45.0, 45.0, 45.0]]]], dtype=np.float32),
    )


@pytest.mark.parametrize("suffix", [".npy", ".raw"])
def test_user_tensor_is_opaque_and_backend_ready(
    hailo_exporter_module: ModuleType, tmp_path: Path, suffix: str
) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    source = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    path = calibration_dir / f"model-domain{suffix}"
    if suffix == ".npy":
        np.save(path, source)
    else:
        source.tofile(path)
    exporter, _ = _externalized_exporter(
        hailo_exporter_module,
        tmp_path,
        calibration_dir,
        shape=[1, 3, 2, 2],
        encoding={"from": "RGB", "to": "BGR"},
        mean_values=[10, 20, 30],
    )

    actual = exporter._get_calibration_data(_Runner([1, 2, 2, 3]))[
        "hailo_input"
    ][0]

    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize("suffix", [".npy", ".raw"])
def test_user_tensor_with_wrong_shape_is_rejected(
    hailo_exporter_module: ModuleType, tmp_path: Path, suffix: str
) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    source = np.arange(6, dtype=np.float32)
    path = calibration_dir / f"wrong-shape{suffix}"
    if suffix == ".npy":
        np.save(path, source.reshape(1, 2, 3))
    else:
        source.tofile(path)
    exporter, _ = _externalized_exporter(
        hailo_exporter_module,
        tmp_path,
        calibration_dir,
        shape=[1, 3, 2, 2],
    )

    with pytest.raises(ModelconverterException, match="expected"):
        exporter._get_calibration_data(_Runner([1, 2, 2, 3]))


def test_generated_tensor_reorders_even_when_axis_sizes_are_equal(
    hailo_exporter_module: ModuleType, tmp_path: Path
) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    source = np.arange(27, dtype=np.float32).reshape(1, 3, 3, 3)
    np.save(calibration_dir / "sample.npy", source)
    exporter, inp = _externalized_exporter(
        hailo_exporter_module,
        tmp_path,
        calibration_dir,
        shape=[1, 3, 3, 3],
        mean_values=[1, 2, 3],
    )
    calibration = inp.calibration
    assert isinstance(calibration, ImageCalibrationConfig)
    calibration._generated_from_random = True

    actual = exporter._get_calibration_data(_Runner([1, 3, 3, 3]))[
        "hailo_input"
    ][0]

    means = np.array([1, 2, 3], dtype=np.float32).reshape(1, 3, 1, 1)
    expected = ((source - means) / 2).transpose(0, 2, 3, 1)[0]
    np.testing.assert_array_equal(actual, expected)


def test_disabled_calibration_accepts_archive_preprocessing_retry(
    hailo_exporter_module: ModuleType, tmp_path: Path
) -> None:
    model = single_io_onnx(tmp_path / "model.onnx").resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": [1, 3, 64, 64],
            "encoding": "BGR",
            "mean_values": [1, 2, 3],
            "onnx_simplification": False,
            "hailo.disable_calibration": True,
        },
    )
    retry_config = config.model_copy(deep=True)
    first_output = tmp_path / "first"
    first_output.mkdir()

    with pytest.raises(
        PreprocessingEmbeddingError, match=r"calibration.*disabled"
    ):
        hailo_exporter_module.HailoExporter(
            next(iter(config.stages.values())), first_output
        )

    extract_preprocessing(retry_config)
    retry_output = tmp_path / "retry"
    retry_output.mkdir()
    exporter = hailo_exporter_module.HailoExporter(
        next(iter(retry_config.stages.values())), retry_output
    )

    assert exporter._disable_calibration
    assert not exporter.inputs["input0"].requires_input_preprocessing()
