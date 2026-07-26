"""Shared helpers for the host-side conversion tests."""

from pathlib import Path

import cv2
import numpy as np
import yaml

from modelconverter.utils.constants import OUTPUTS_DIR
from tests.helpers.onnx_factory import build_toy_conv_onnx

__all__ = [
    "HAILO_FAST_OPTS",
    "assert_produced",
    "toy_net_image_inputs",
    "write_toy_conv_config",
]

# Cheap Hailo settings for a smoke conversion: skip the slow HEF compile and
# drop optimization/compression so quantizing a tiny net stays fast.
HAILO_FAST_OPTS = (
    "hailo.compression_level",
    "0",
    "hailo.optimization_level",
    "0",
    "hailo.disable_compilation",
    "True",
)


def assert_produced(output_name: str, to_format: str = "native") -> None:
    """Assert a conversion wrote an artifact into its output dir."""
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    if to_format == "nn_archive":
        produced = list(out_dir.rglob("*.tar.xz")) + list(
            out_dir.rglob("*.tar")
        )
        assert produced, f"no NN archive produced in {out_dir}"
    else:
        assert any(out_dir.iterdir()), f"no output produced in {out_dir}"


def toy_net_image_inputs(calib: dict, *, size: int = 64) -> list[dict]:
    """The toy integration net's three image inputs (``bgr``/``rgb``/``gray``)
    with their per-input mean/scale/encoding, all sharing one ``calib`` spec.

    Shared by the toy-integration and toy-multistage conversion tests, which
    differ only in the calibration (random constant vs. an image directory).
    """
    return [
        {
            "name": "bgr",
            "shape": [1, 3, size, size],
            "data_type": "float32",
            "mean_values": [10, 20, 30],
            "scale_values": [2, 4, 8],
            "encoding": {"from": "BGR", "to": "BGR"},
            "calibration": calib,
        },
        {
            "name": "rgb",
            "shape": [1, 3, size, size],
            "data_type": "float32",
            "mean_values": [5, 6, 7],
            "scale_values": [3, 2, 1],
            "encoding": {"from": "RGB", "to": "BGR"},
            "calibration": calib,
        },
        {
            "name": "gray",
            "shape": [1, 1, size, size],
            "data_type": "float32",
            "mean_values": [128],
            "scale_values": [255],
            "encoding": "GRAY",
            "calibration": calib,
        },
    ]


def write_toy_conv_config(
    workdir: Path,
    *,
    size: int = 32,
    out_channels: int = 4,
    external_data: bool = False,
) -> Path:
    """Build the toy conv ONNX + a 16-image calibration dir + a matching
    conversion config, returning the config's yaml path.

    Shared by the single-input conversion tests (precision, external-data,
    hailo-compile, rvc4-encodings). The config carries mean/scale + an
    RGB->BGR reversal so the converted model exercises the full baked
    preprocessing path; the calibration images are random so the per-tensor
    quantization ranges stay healthy.
    """
    onnx_path = workdir / "toy_conv.onnx"
    build_toy_conv_onnx(
        onnx_path,
        size=size,
        out_channels=out_channels,
        external_data=external_data,
    )

    calib_dir = workdir / "calib"
    calib_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(16):
        cv2.imwrite(
            str(calib_dir / f"{i}.png"),
            rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        )

    config = {
        "input_model": str(onnx_path),
        "inputs": [
            {
                "name": "img",
                "shape": [1, 3, size, size],
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
