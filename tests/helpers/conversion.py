"""Shared helpers for the host-side conversion tests."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml

from modelconverter.utils.constants import OUTPUTS_DIR
from tests.helpers.onnx_factory import build_toy_conv_onnx

# Skip the slow HEF compile and all optimization -- enough for a conversion
# smoke check, and for a fidelity check run on the quantized HAR.
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


def assert_produced_suffix(output_name: str, suffix: str) -> None:
    """Assert a conversion produced non-empty ``suffix`` files.

    Prefers matches directly in the output dir over nested ones, so the final
    artifact wins over the ``intermediate_outputs/`` copies.
    """
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    produced = list(out_dir.glob(f"*{suffix}")) or list(
        out_dir.rglob(f"*{suffix}")
    )
    assert produced, f"no {suffix} produced in {out_dir}"
    assert all(p.stat().st_size > 0 for p in produced), (
        f"empty {suffix} in {out_dir}"
    )


def toy_net_image_inputs(calib: dict, *, size: int = 64) -> list[dict]:
    """The toy integration net's ``bgr``/``rgb``/``gray`` inputs, each with its
    own mean/scale/encoding and all sharing one ``calib`` spec.

    Shared by the toy-integration and toy-multistage tests, which differ only in
    the calibration (a random constant vs. an image directory).
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


def write_config(workdir: Path, name: str, config: dict) -> Path:
    """Write a conversion config as yaml, returning its path."""
    path = workdir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _write_calibration(
    calib_dir: Path, kind: Literal["png", "raw", "npy"], size: int
) -> None:
    calib_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(16):
        if kind == "raw":
            # SNPE reads `.raw` files verbatim, so write one correctly-sized
            # NHWC float32 tensor per file.
            rng.uniform(0, 255, (size, size, 3)).astype(np.float32).tofile(
                calib_dir / f"{i}.raw"
            )
        elif kind == "npy":
            # A `.npy` is loaded verbatim (no image decode), so a pre-shaped
            # `(1, C, H, W)` tensor is what reaches hailo's NCHW->NHWC
            # calibration transpose.
            np.save(
                calib_dir / f"{i}.npy",
                rng.uniform(0, 255, (1, 3, size, size)).astype(np.float32),
            )
        else:
            cv2.imwrite(
                str(calib_dir / f"{i}.png"),
                rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
            )


def write_toy_conv_config(
    workdir: Path,
    *,
    size: int = 32,
    out_channels: int = 4,
    external_data: bool = False,
    calibration: Literal["png", "raw", "npy", "none"] = "png",
) -> Path:
    """Build the toy conv ONNX plus a matching config, returning the yaml path.

    Shared by the single-input conversion tests (precision, external-data,
    hailo-compile, rvc4-encodings). The config carries mean/scale and an
    RGB->BGR reversal, so the converted model gets the full baked preprocessing.

    ``calibration="none"`` leaves the input to fall back to random calibration
    (for a ``disable_calibration`` run).
    """
    onnx_path = workdir / "toy_conv.onnx"
    build_toy_conv_onnx(
        onnx_path,
        size=size,
        out_channels=out_channels,
        external_data=external_data,
    )

    input_cfg: dict = {
        "name": "img",
        "shape": [1, 3, size, size],
        "data_type": "float32",
        "mean_values": [128, 128, 128],
        "scale_values": [58, 58, 58],
        "encoding": {"from": "RGB", "to": "BGR"},
    }
    if calibration != "none":
        calib_dir = workdir / "calib"
        _write_calibration(calib_dir, calibration, size)
        input_cfg["calibration"] = {"path": str(calib_dir), "max_images": 16}

    return write_config(
        workdir,
        "toy_conv",
        {
            "input_model": str(onnx_path),
            "inputs": [input_cfg],
            "outputs": [{"name": "out"}],
        },
    )
