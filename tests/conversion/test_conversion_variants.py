"""Conversions of less-common input / calibration variants.

Each converts a tiny toy model on one platform, covering an exporter branch the
main conversion tests miss -- an unusual calibration format, or an input shape
other than the usual 4D ``NCHW``.

Run inside the platform Docker image, e.g.::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k raw_calibration'
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from modelconverter.__main__ import convert
from modelconverter.utils.types import Platform
from tests.helpers.conversion import (
    HAILO_FAST_OPTS,
    assert_produced,
    write_config,
    write_toy_conv_config,
)
from tests.helpers.onnx_factory import build_relu_onnx
from tests.helpers.platform_options import platform_options
from tests.helpers.tflite_factory import build_toy_tflite


@pytest.mark.rvc4
def test_rvc4_raw_calibration(tmp_path: Path):
    # SNPE consumes `.raw` calibration files verbatim, bypassing the image
    # reader every other test goes through.
    config = write_toy_conv_config(tmp_path, calibration="raw")
    output_name = "_rvc4-raw-calib"
    convert(
        Platform.RVC4, path=str(config), output_dir=output_name, to="native"
    )
    assert_produced(output_name)


@pytest.mark.hailo
def test_hailo_without_calibration(tmp_path: Path):
    # No calibration data at all; `disable_calibration` returns the float HAR.
    config = write_toy_conv_config(tmp_path, calibration="none")
    output_name = "_hailo-no-calib"
    convert(
        Platform.HAILO,
        "hailo.disable_calibration",
        "True",
        path=str(config),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)


@pytest.mark.rvc2
def test_rvc2_hwc_input(tmp_path: Path):
    # A rank-3 channels-last input rather than the usual 4D NCHW.
    onnx_path = build_relu_onnx(tmp_path / "hwc.onnx", [32, 32, 3])
    config_path = write_config(
        tmp_path,
        "hwc",
        {
            "input_model": str(onnx_path),
            "inputs": [
                {
                    "name": "data",
                    "shape": [32, 32, 3],
                    "layout": "HWC",
                    "data_type": "float32",
                    # Raw input: no baked preprocessing, so the model optimizer
                    # need not disambiguate this 3D input's channel axis.
                    "encoding": "NONE",
                }
            ],
            "outputs": [{"name": "out"}],
        },
    )
    output_name = "_rvc2-hwc"
    convert(
        Platform.RVC2,
        *platform_options(Platform.RVC2),
        path=str(config_path),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)


@pytest.mark.rvc2
def test_rvc2_tflite_hwc_input(tmp_path: Path):
    # A rank-3 `HWC` TFLite input drives the `chw->hwc` reshape in
    # `_transform_tflite_to_onnx`.
    tflite_path = build_toy_tflite(tmp_path / "hwc.tflite", shape=(8, 8, 3))
    output_name = "_rvc2-tflite-hwc"
    convert(
        Platform.RVC2,
        *platform_options(Platform.RVC2),
        path=str(tflite_path),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)


@pytest.mark.rvc4
def test_rvc4_ncd_layout(tmp_path: Path):
    # A rank-3 `[1, C, D]` input gets the fallback `NCD` layout, which SNPE
    # rewrites to `NCF`. That happens in `_onnx_to_dlc`, before and independently
    # of quantization, so calibration is disabled -- a non-image `[1, 4, 5]`
    # shape has no meaningful random-image calibration anyway.
    onnx_path = build_relu_onnx(tmp_path / "ncd.onnx", [1, 4, 5])
    output_name = "_rvc4-ncd"
    convert(
        Platform.RVC4,
        "rvc4.disable_calibration",
        "True",
        path=str(onnx_path),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)


@pytest.mark.hailo
def test_hailo_tflite_conversion(tmp_path: Path):
    # A full hailo conversion from TFLite exercises the `is_tflite` channel-axis
    # branch in `_get_alls`. The toy graph is a weightless passthrough, on which
    # hailo's quantizer degenerates (0 dB SNR -> NaN params) if a `bgr_to_rgb`
    # layer is inserted, so the config keeps `from == to` and uses fixed image
    # calibration -- leaving a normalization + Relu that quantizes
    # deterministically.
    tflite_path = build_toy_tflite(tmp_path / "toy.tflite")  # [1, 8, 8, 3]
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        cv2.imwrite(
            str(calib_dir / f"{i}.png"),
            rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
        )
    config_path = write_config(
        tmp_path,
        "toy_tflite",
        {
            "input_model": str(tflite_path),
            "inputs": [
                {
                    "name": "input",
                    "shape": [1, 8, 8, 3],
                    "layout": "NHWC",
                    "data_type": "float32",
                    "encoding": {"from": "RGB", "to": "RGB"},
                    "calibration": {"path": str(calib_dir), "max_images": 8},
                }
            ],
            "outputs": [{"name": "output"}],
        },
    )
    output_name = "_hailo-tflite"
    convert(
        Platform.HAILO,
        *HAILO_FAST_OPTS,
        path=str(config_path),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)


@pytest.mark.hailo
def test_hailo_npy_calibration(tmp_path: Path):
    # `.npy` calibration files are pre-shaped `(1, C, H, W)` tensors, which
    # hailo's calibration reader loads verbatim and transposes to NHWC -- a
    # branch `.png` calibration (read as `HWC`) never reaches.
    config = write_toy_conv_config(tmp_path, calibration="npy")
    output_name = "_hailo-npy-calib"
    convert(
        Platform.HAILO,
        *HAILO_FAST_OPTS,
        path=str(config),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)
