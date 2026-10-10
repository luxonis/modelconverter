"""Numeric contract tests for externalized calibration preprocessing."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modelconverter.cli.utils import extract_preprocessing
from modelconverter.platforms.base_exporter import Exporter
from modelconverter.utils.config import (
    Config,
    ImageCalibrationConfig,
    InputConfig,
)
from modelconverter.utils.exceptions import ModelconverterException
from modelconverter.utils.preprocessing import (
    CalibrationPreprocessing,
    apply_calibration_preprocessing,
    channels_last_4d_layout,
    channels_last_image_layout,
    read_user_calibration_tensor,
    reorder_layout,
)
from modelconverter.utils.types import DataType, Encoding


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_color_conversion_precedes_channelwise_normalization(layout: str):
    # Reversed to [10, 20, 30], then (x - mean) / scale per channel.
    shape = (1, 3, 1, 1) if layout == "NCHW" else (1, 1, 1, 3)
    bgr = np.array([30, 20, 10], dtype=np.float32).reshape(shape)
    expected = np.array([2.25, 3.6, 4.5], dtype=np.float32).reshape(shape)

    preprocessing = CalibrationPreprocessing(
        encoding_from=Encoding.RGB,
        encoding_to=Encoding.BGR,
        mean_values=(1.0, 2.0, 3.0),
        scale_values=(4.0, 5.0, 6.0),
        data_type=DataType.FLOAT32,
        is_image=True,
    )

    actual = apply_calibration_preprocessing(bgr, preprocessing, layout=layout)

    np.testing.assert_allclose(actual, expected)


def test_scalar_normalization_broadcasts_and_preserves_negative_values():
    array = np.array([0.0, 10.0, 20.0], dtype=np.float32).reshape(3, 1, 1)
    preprocessing = CalibrationPreprocessing(
        encoding_from=Encoding.NONE,
        encoding_to=Encoding.NONE,
        mean_values=(10.0,),
        scale_values=(2.0,),
        data_type=DataType.FLOAT32,
        is_image=False,
    )

    actual = apply_calibration_preprocessing(
        array, preprocessing, layout="CHW"
    )

    np.testing.assert_array_equal(
        actual[:, 0, 0], np.array([-5.0, 0.0, 5.0], dtype=np.float32)
    )


def test_normalizing_to_integer_model_input_is_rejected():
    preprocessing = CalibrationPreprocessing(
        encoding_from=Encoding.NONE,
        encoding_to=Encoding.NONE,
        mean_values=(1.0,),
        scale_values=(2.0,),
        data_type=DataType.UINT8,
        is_image=False,
    )

    with pytest.raises(ModelconverterException, match="floating-point"):
        apply_calibration_preprocessing(
            np.ones((1, 2), dtype=np.uint8),
            preprocessing,
            layout="NC",
        )


def test_normalizing_without_channel_axis_is_rejected():
    preprocessing = CalibrationPreprocessing(
        encoding_from=Encoding.NONE,
        encoding_to=Encoding.NONE,
        mean_values=(1.0,),
        scale_values=(2.0,),
        data_type=DataType.FLOAT32,
        is_image=False,
    )

    with pytest.raises(ModelconverterException, match="channel axis"):
        apply_calibration_preprocessing(
            np.ones((1, 2), dtype=np.float32), preprocessing, layout="NF"
        )


def _preprocessing(
    *, reverse: bool = False, mean_values: tuple[float, ...] | None = None
) -> CalibrationPreprocessing:
    """Preprocessing of an RGB float image input."""
    return CalibrationPreprocessing(
        encoding_from=Encoding.RGB,
        encoding_to=Encoding.BGR if reverse else Encoding.RGB,
        mean_values=mean_values,
        scale_values=None,
        data_type=DataType.FLOAT32,
        is_image=True,
    )


def test_layout_must_name_every_array_axis():
    with pytest.raises(ModelconverterException, match="cannot use layout"):
        apply_calibration_preprocessing(
            np.ones((3, 2, 2)), _preprocessing(), layout="NCHW"
        )


def test_channel_reversal_requires_three_channels():
    with pytest.raises(ModelconverterException, match="exactly three"):
        apply_calibration_preprocessing(
            np.ones((1, 4, 2, 2)),
            _preprocessing(reverse=True),
            layout="NCHW",
        )


def test_mean_without_scale_only_subtracts():
    array = np.full((1, 3, 1, 1), 5, dtype=np.float32)

    result = apply_calibration_preprocessing(
        array, _preprocessing(mean_values=(1.0, 2.0, 3.0)), layout="NCHW"
    )

    np.testing.assert_array_equal(result.ravel(), [4.0, 3.0, 2.0])


def test_mean_needs_one_value_or_one_per_channel():
    with pytest.raises(ModelconverterException, match="has 2 values"):
        apply_calibration_preprocessing(
            np.ones((1, 3, 1, 1), dtype=np.float32),
            _preprocessing(mean_values=(1.0, 2.0)),
            layout="NCHW",
        )


def test_raw_user_tensor_needs_a_shape(tmp_path: Path):
    path = tmp_path / "sample.raw"
    np.zeros(4, dtype=np.float32).tofile(path)

    with pytest.raises(ValueError, match="shape is required"):
        read_user_calibration_tensor(
            path, data_type=DataType.FLOAT32, input_name="input0"
        )


def test_reorder_layout_adds_and_removes_only_singleton_batch():
    hwc = np.arange(24).reshape(2, 4, 3)
    nchw = reorder_layout(hwc, "HWC", "NCHW")

    assert nchw.shape == (1, 3, 2, 4)
    np.testing.assert_array_equal(reorder_layout(nchw, "NCHW", "HWC"), hwc)


def test_reorder_layout_keeps_a_batch_of_two():
    with pytest.raises(ModelconverterException, match="singleton"):
        reorder_layout(np.ones((2, 4, 4, 3)), "NHWC", "HWC")


def test_reorder_layout_needs_the_same_axes():
    with pytest.raises(ModelconverterException, match="'HWC' to 'HWD'"):
        reorder_layout(np.ones((4, 4, 3)), "HWC", "HWD")


def test_channels_last_image_layout_keeps_a_non_image_layout():
    assert channels_last_image_layout("NCD") == "NCD"


def test_channels_last_4d_layout_keeps_a_3d_layout():
    assert channels_last_4d_layout("CHW") == "CHW"


def _externalized_input(
    dummy_onnx: Path,
    calibration_dir: Path,
) -> InputConfig:
    cfg = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "shape": [1, 3, 1, 1],
            "layout": "NCHW",
            "data_type": "float32",
            "encoding": {"from": "RGB", "to": "BGR"},
            "mean_values": [10, 20, 30],
            "scale_values": 2,
            "calibration": {"path": str(calibration_dir)},
        },
    )
    extract_preprocessing(cfg)
    return next(iter(cfg.stages.values())).inputs[0]


def test_externalized_image_is_preprocessed(dummy_onnx: Path, tmp_path: Path):
    calibration_dir = tmp_path / "images"
    calibration_dir.mkdir()
    image_path = calibration_dir / "pixel.png"
    Image.fromarray(np.array([[[100, 110, 120]]], dtype=np.uint8)).save(
        image_path
    )
    inp = _externalized_input(dummy_onnx, calibration_dir)
    calib = inp.calibration
    assert isinstance(calib, ImageCalibrationConfig)

    array, layout = Exporter._read_calibration_file(
        inp,
        calib,
        image_path,
    )

    assert layout == "HWC"
    np.testing.assert_array_equal(
        array, np.array([[[45.0, 45.0, 45.0]]], dtype=np.float32)
    )


@pytest.mark.parametrize("suffix", [".npy", ".raw"])
def test_user_tensor_reader_preserves_values_and_reshapes_raw(
    tmp_path: Path, suffix: str
):
    calibration_dir = tmp_path / suffix.lstrip(".")
    calibration_dir.mkdir()
    source = np.array([[[[100.0]], [[110.0]], [[120.0]]]], dtype=np.float32)
    tensor_path = calibration_dir / f"sample{suffix}"
    if suffix == ".npy":
        np.save(tensor_path, source)
    else:
        source.tofile(tensor_path)
    array = read_user_calibration_tensor(
        tensor_path,
        raw_shape=source.shape,
        data_type=DataType.FLOAT32,
        input_name="input0",
    )

    np.testing.assert_array_equal(array, source)


def test_user_numpy_tensor_does_not_infer_or_validate_source_layout(
    tmp_path: Path,
):
    calibration_dir = tmp_path / "rank-two"
    calibration_dir.mkdir()
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    tensor_path = calibration_dir / "sample.npy"
    np.save(tensor_path, source)
    array = read_user_calibration_tensor(
        tensor_path,
        data_type=DataType.FLOAT32,
        input_name="input0",
    )

    np.testing.assert_array_equal(array, source)


def test_externalized_grayscale_image_keeps_single_channel(
    dummy_onnx: Path, tmp_path: Path
):
    calibration_dir = tmp_path / "grayscale"
    calibration_dir.mkdir()
    image_path = calibration_dir / "pixel.png"
    Image.fromarray(np.array([[100]], dtype=np.uint8)).save(image_path)
    cfg = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "shape": [1, 1, 1, 1],
            "layout": "NCHW",
            "data_type": "float32",
            "encoding": "GRAY",
            "mean_values": 10,
            "scale_values": 2,
            "calibration": {"path": str(calibration_dir)},
        },
    )
    extract_preprocessing(cfg)
    inp = next(iter(cfg.stages.values())).inputs[0]
    calib = inp.calibration
    assert isinstance(calib, ImageCalibrationConfig)

    array, layout = Exporter._read_calibration_file(
        inp,
        calib,
        image_path,
    )

    assert layout == "HWC"
    assert array.shape == (1, 1, 1)
    np.testing.assert_array_equal(
        array, np.array([[[45.0]]], dtype=np.float32)
    )


def test_generated_numpy_calibration_is_managed(
    dummy_onnx: Path, tmp_path: Path
):
    calibration_dir = tmp_path / "generated"
    calibration_dir.mkdir()
    # Generated data follows the configured runtime encoding (BGR here).
    source = np.array([[[[120.0]], [[110.0]], [[100.0]]]], dtype=np.float32)
    npy_path = calibration_dir / "sample.npy"
    np.save(npy_path, source)
    inp = _externalized_input(dummy_onnx, calibration_dir)
    calib = inp.calibration
    assert isinstance(calib, ImageCalibrationConfig)
    calib._generated_from_random = True
    assert inp.layout is not None
    calib._generated_layout = inp.layout

    array, layout = Exporter._read_calibration_file(
        inp,
        calib,
        npy_path,
    )

    assert layout == "NCHW"
    np.testing.assert_array_equal(
        array,
        np.array([[[[45.0]], [[45.0]], [[45.0]]]], dtype=np.float32),
    )
