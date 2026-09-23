"""Host-side tests for Hailo calibration tensor preparation."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, TypedDict

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
    RandomCalibrationConfig,
)
from tests.helpers.onnx_factory import single_io_onnx

if TYPE_CHECKING:
    from modelconverter.platforms.hailo.exporter import HailoExporter


class _HnLayer(TypedDict):
    """HN layer fields read by `HailoExporter._get_hn_layer_info`."""

    original_names: list[str]
    input_shapes: list[list[int]]


class _HnDict(TypedDict):
    """The portion of the Hailo network dictionary used by these tests."""

    layers: dict[str, _HnLayer]


class _Runner:
    def __init__(self, input_shape: list[int] | None = None) -> None:
        self.input_shape = input_shape or [1, 1, 1, 3]

    def get_hn_dict(self) -> _HnDict:
        return {
            "layers": {
                "hailo_input": {
                    "original_names": ["input0"],
                    "input_shapes": [self.input_shape],
                }
            }
        }


class _FakeHailoSdk(ModuleType):
    ClientRunner = object
    __version__ = "test"


class _FakeTensorflow(ModuleType):
    config = SimpleNamespace(list_physical_devices=lambda _kind: [])


def _load_hailo_exporter_module(
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    """Import the exporter with fake SDK modules and restorable module state."""
    monkeypatch.setitem(
        sys.modules, "hailo_sdk_client", _FakeHailoSdk("hailo_sdk_client")
    )
    monkeypatch.setitem(
        sys.modules, "tensorflow", _FakeTensorflow("tensorflow")
    )
    package = importlib.import_module("modelconverter.platforms.hailo")
    module_name = "modelconverter.platforms.hailo.exporter"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        package,
        "exporter",
        getattr(package, "exporter", None),
        raising=False,
    )

    return importlib.import_module(module_name)


@pytest.fixture
def hailo_exporter_module(
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    return _load_hailo_exporter_module(monkeypatch)


def _externalized_exporter(
    hailo_exporter_module: ModuleType,
    tmp_path: Path,
    calibration_dir: Path,
    *,
    shape: list[int] | None = None,
    encoding: str | dict[str, str] = "RGB",
    mean_values: list[int] | int = 0,
) -> tuple["HailoExporter", InputConfig]:
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
    exporter: HailoExporter = object.__new__(
        hailo_exporter_module.HailoExporter
    )
    exporter._inputs = {inp.name: inp for inp in stage.inputs}
    return exporter, stage.inputs[0]


def test_exporter_import_restores_preexisting_module_and_package_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "modelconverter.platforms.hailo.exporter"
    package = importlib.import_module("modelconverter.platforms.hailo")
    preexisting = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, preexisting)
    monkeypatch.setattr(package, "exporter", preexisting, raising=False)

    with pytest.MonkeyPatch.context() as isolated:
        loaded = _load_hailo_exporter_module(isolated)
        assert loaded is not preexisting
        assert package.exporter is loaded

    assert sys.modules[module_name] is preexisting
    assert package.exporter is preexisting


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
    source = np.arange(60, dtype=np.float32).reshape(4, 5, 3)
    path = calibration_dir / f"model-domain{suffix}"
    if suffix == ".npy":
        np.save(path, source)
    else:
        source.tofile(path)
    exporter, _ = _externalized_exporter(
        hailo_exporter_module,
        tmp_path,
        calibration_dir,
        shape=[1, 3, 4, 5],
        encoding={"from": "RGB", "to": "BGR"},
        mean_values=[10, 20, 30],
    )

    actual = exporter._get_calibration_data(_Runner([1, 4, 5, 3]))[
        "hailo_input"
    ][0]

    np.testing.assert_array_equal(actual, source)


def test_user_npy_with_singleton_batch_is_accepted(
    hailo_exporter_module: ModuleType, tmp_path: Path
) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    source = np.arange(60, dtype=np.float32).reshape(4, 5, 3)
    np.save(calibration_dir / "sample.npy", source[np.newaxis])
    exporter, _ = _externalized_exporter(
        hailo_exporter_module,
        tmp_path,
        calibration_dir,
        shape=[1, 3, 4, 5],
    )

    actual = exporter._get_calibration_data(_Runner([1, 4, 5, 3]))[
        "hailo_input"
    ][0]

    np.testing.assert_array_equal(actual, source)


def test_user_nchw_npy_is_rejected_without_implicit_reordering(
    hailo_exporter_module: ModuleType, tmp_path: Path
) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    source = np.arange(60, dtype=np.float32).reshape(4, 5, 3)
    np.save(calibration_dir / "sample.npy", source.transpose(2, 0, 1)[None])
    exporter, _ = _externalized_exporter(
        hailo_exporter_module,
        tmp_path,
        calibration_dir,
        shape=[1, 3, 4, 5],
    )

    with pytest.raises(ModelconverterException, match="expected"):
        exporter._get_calibration_data(_Runner([1, 4, 5, 3]))


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
    assert inp.layout == "NCHW"
    calibration._generated_from_random = True
    calibration._generated_layout = inp.layout

    actual = exporter._get_calibration_data(_Runner([1, 3, 3, 3]))[
        "hailo_input"
    ][0]

    means = np.array([1, 2, 3], dtype=np.float32).reshape(1, 3, 1, 1)
    expected = ((source - means) / 2).transpose(0, 2, 3, 1)[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("channels", [256, 20])
def test_generated_non_image_tensor_uses_hailo_channel_last_layout(
    hailo_exporter_module: ModuleType, tmp_path: Path, channels: int
) -> None:
    inp = InputConfig.model_validate(
        {
            "name": "input0",
            "shape": [1, channels, 20, 20],
            "encoding": "NONE",
        }
    )
    inp.calibration = RandomCalibrationConfig(max_images=1)
    assert inp.layout == "NCDE"
    exporter: HailoExporter = object.__new__(
        hailo_exporter_module.HailoExporter
    )
    exporter._inputs = {inp.name: inp}
    exporter.intermediate_outputs_dir = tmp_path
    exporter._prepare_random_calibration_data()

    calibration = inp.calibration
    assert isinstance(calibration, ImageCalibrationConfig)
    generated = np.load(calibration.path / "0.npy")
    actual = exporter._get_calibration_data(_Runner([1, 20, 20, channels]))[
        "hailo_input"
    ][0]

    np.testing.assert_array_equal(actual, generated.transpose(0, 2, 3, 1)[0])


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
    exporter: HailoExporter = hailo_exporter_module.HailoExporter(
        next(iter(retry_config.stages.values())), retry_output
    )

    assert exporter._disable_calibration
    assert not exporter.inputs["input0"].requires_input_preprocessing()
