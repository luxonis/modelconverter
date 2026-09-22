"""RVC3 calibration tests that do not require the OpenVINO toolchain."""

import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pytest
from luxonis_ml.typing import Params
from PIL import Image

from modelconverter.cli.utils import extract_preprocessing
from modelconverter.platforms.rvc3 import exporter as rvc3_exporter
from modelconverter.platforms.rvc3.exporter import RVC3Exporter
from modelconverter.utils import ModelconverterException
from modelconverter.utils.config import (
    Config,
    EncodingConfig,
    ImageCalibrationConfig,
    InputConfig,
)
from modelconverter.utils.types import DataType, Encoding
from tests.helpers.onnx_factory import single_io_onnx


def test_externalized_preprocessing_uses_float_numpy_pot_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    Image.fromarray(np.array([[[100, 110, 120]]], dtype=np.uint8)).save(
        calibration_dir / "pixel.png"
    )
    exporter = _calibrating_exporter(
        tmp_path,
        [1, 3, 1, 1],
        {
            "layout": "NCHW",
            "encoding": {"from": "RGB", "to": "BGR"},
            "mean_values": [10, 20, 30],
            "scale_values": 2,
            "calibration": {"path": str(calibration_dir)},
        },
    )
    commands: list[list[object]] = []
    monkeypatch.setattr(
        rvc3_exporter,
        "subprocess_run",
        lambda command, **_kwargs: commands.append(command),
    )

    exporter._calibrate(tmp_path / "model.xml")

    dataset = _pot_dataset(exporter)
    assert dataset.reader == "numpy_reader"
    actual = np.load(dataset.data_source / "0.npy")
    assert actual.dtype == np.float32
    assert actual.shape == (1, 1, 3)
    np.testing.assert_array_equal(
        actual,
        np.array([[[45.0, 45.0, 45.0]]], dtype=np.float32),
    )
    # POT's Accuracy Checker batches HWC NumPy samples and converts the batch
    # to the OpenVINO model layout.
    pot_input = np.expand_dims(actual, axis=0).transpose(0, 3, 1, 2)
    assert pot_input.shape == (1, 3, 1, 1)
    assert commands
    assert commands[0][0] == "pot"


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [
        (Encoding.BGR, [0, 0, 255]),
        (Encoding.RGB, [255, 0, 0]),
    ],
)
def test_image_and_numpy_pot_readers_keep_the_same_color_order(
    tmp_path: Path, encoding: Encoding, expected: list[int]
) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    image_path = calibration_dir / "red.png"
    Image.fromarray(np.full((2, 2, 3), (255, 0, 0), np.uint8)).save(image_path)
    inp = InputConfig(
        name="input0",
        shape=[1, 3, 2, 2],
        layout="NCHW",
        encoding=EncodingConfig.model_validate(
            {"from": encoding, "to": encoding}
        ),
        data_type=DataType.UINT8,
    )
    calib = ImageCalibrationConfig(path=calibration_dir)
    exporter = object.__new__(RVC3Exporter)
    exporter.intermediate_outputs_dir = tmp_path

    image_dataset = exporter._write_calibration_images(
        inp, calib, [image_path], shape=[1, 3, 2, 2]
    )
    data_source = image_dataset["data_source"]
    assert isinstance(data_source, str)
    image_sample = cv2.imread(str(Path(data_source) / image_path.name))
    exporter._write_calibration_tensors(
        inp, calib, [image_path], shape=[1, 3, 2, 2]
    )
    numpy_sample = np.load(tmp_path / "calibration_tensors/0.npy")

    assert "preprocessing" not in image_dataset
    np.testing.assert_array_equal(image_sample[0, 0], expected)
    np.testing.assert_array_equal(numpy_sample[0, 0], expected)


@pytest.mark.parametrize("suffix", [".npy", ".raw"])
def test_user_tensor_is_opaque_but_serialized_for_pot_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    shape = [1, 3, 1, 2]
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    source = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    tensor_path = calibration_dir / f"sample{suffix}"
    if suffix == ".npy":
        np.save(tensor_path, source)
    else:
        source.tofile(tensor_path)
    exporter = _calibrating_exporter(
        tmp_path,
        shape,
        {
            "layout": "NCHW",
            "encoding": {"from": "RGB", "to": "BGR"},
            "mean_values": [100, 200, 300],
            "scale_values": 10,
            "calibration": {"path": str(calibration_dir)},
        },
    )
    monkeypatch.setattr(
        rvc3_exporter, "subprocess_run", lambda *_args, **_kwargs: None
    )

    exporter._calibrate(tmp_path / "model.xml")

    dataset = _pot_dataset(exporter)
    assert dataset.reader == "numpy_reader"
    actual = np.load(dataset.data_source / "0.npy")
    np.testing.assert_array_equal(actual, source)


def test_user_tensor_uses_generic_four_dimensional_channel_last_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = [1, 19, 7, 8]
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    source = np.arange(7 * 8 * 19, dtype=np.float32).reshape(7, 8, 19)
    np.save(calibration_dir / "sample.npy", source)
    exporter = _calibrating_exporter(
        tmp_path,
        shape,
        {
            "layout": "NCDE",
            "encoding": "NONE",
            "calibration": {"path": str(calibration_dir)},
        },
    )
    monkeypatch.setattr(
        rvc3_exporter, "subprocess_run", lambda *_args, **_kwargs: None
    )

    exporter._calibrate(tmp_path / "model.xml")

    actual = np.load(_pot_dataset(exporter).data_source / "0.npy")
    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize("suffix", [".npy", ".raw"])
def test_user_tensor_with_wrong_shape_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    source = np.arange(5, dtype=np.float32)
    tensor_path = calibration_dir / f"wrong-shape{suffix}"
    if suffix == ".npy":
        np.save(tensor_path, source.reshape(1, 5))
    else:
        source.tofile(tensor_path)
    exporter = _calibrating_exporter(
        tmp_path,
        [1, 3, 1, 2],
        {
            "layout": "NCHW",
            "encoding": "RGB",
            "calibration": {"path": str(calibration_dir)},
        },
    )
    monkeypatch.setattr(
        rvc3_exporter, "subprocess_run", lambda *_args, **_kwargs: None
    )

    with pytest.raises(ModelconverterException, match="expected"):
        exporter._calibrate(tmp_path / "model.xml")


def test_generated_calibration_keeps_layout_across_tflite_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated NHWC tensors remain NHWC until the converted model reorders them."""
    exporter = _calibrating_exporter(
        tmp_path,
        [1, 2, 4, 3],
        {
            "layout": "NHWC",
            "encoding": "RGB",
            "mean_values": [1, 2, 3],
            "scale_values": 2,
            "calibration": {"max_images": 1, "data_type": "float32"},
        },
    )
    inp = exporter.inputs["input0"]
    calibration = inp.calibration
    assert isinstance(calibration, ImageCalibrationConfig)
    source = np.load(next(calibration.path.glob("*.npy")))

    # RVC3's TFLite-to-ONNX conversion changes the model input contract after
    # random calibration has already been generated by Exporter.__init__.
    inp.shape = [1, 3, 2, 4]
    inp.layout = "NCHW"
    monkeypatch.setattr(
        rvc3_exporter, "subprocess_run", lambda *_args, **_kwargs: None
    )

    exporter._calibrate(tmp_path / "model.xml")

    actual = np.load(
        exporter.intermediate_outputs_dir / "calibration_tensors/0.npy"
    )
    expected_model_input = (
        (source - np.array([1, 2, 3], dtype=np.float32).reshape(1, 1, 1, 3))
        / 2
    ).transpose(0, 3, 1, 2)
    assert actual.shape == (2, 4, 3)
    pot_input = np.expand_dims(actual, axis=0).transpose(0, 3, 1, 2)
    assert pot_input.shape == (1, 3, 2, 4)
    np.testing.assert_allclose(pot_input, expected_model_input)


def _calibrating_exporter(
    tmp_path: Path, shape: list[int], input_options: Params
) -> RVC3Exporter:
    """Build an exporter whose preprocessing was moved to the archive."""
    model = single_io_onnx(
        tmp_path / "model.onnx", shape=shape, output_shape=shape
    ).resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": shape,
            "onnx_simplification": False,
            **input_options,
        },
    )
    extract_preprocessing(config)
    output_dir = tmp_path / "rvc3-output"
    output_dir.mkdir()
    return RVC3Exporter(next(iter(config.stages.values())), output_dir)


class _PotDataset(NamedTuple):
    """The parts of POT's dataset description these tests check."""

    reader: str
    data_source: Path


def _pot_dataset(exporter: RVC3Exporter) -> _PotDataset:
    """Read back the dataset POT was configured with."""
    config = json.loads(
        (exporter.intermediate_outputs_dir / "pot_config.json").read_text()
    )
    dataset = config["engine"]["datasets"][0]
    return _PotDataset(dataset["reader"], Path(dataset["data_source"]))
