"""Numeric-fidelity helpers for the conversion precision tests.

A precision test compares a converted model with the original ONNX model
on the same input. The cosine similarity compares the pattern of two
outputs and ignores their scale, which suits a deep network whose
quantization moves the scale a little. The relative error also catches
the right pattern at the wrong scale, for example a lost normalization
in front of a linear model, so the toy models use it.

`record_property` keeps each measured value, and ``tests/conftest.py``
lists the values at the end of the run, so a green run also shows the
margin of each check.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

_MODEL_GLOB = {
    "rvc2": "*.xml",
    "rvc3": "*.xml",
    "rvc4": "*.dlc",
    "hailo": "*.har",
}

RecordProperty = Callable[[str, object], None]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays, compared as flat vectors."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def relative_error(reference: np.ndarray, converted: np.ndarray) -> float:
    """L2 norm of the difference, relative to the L2 norm of the reference."""
    x = np.asarray(reference, dtype=np.float64).ravel()
    y = np.asarray(converted, dtype=np.float64).ravel()
    return float(np.linalg.norm(x - y) / np.linalg.norm(x))


def check_relative_error(
    record_property: RecordProperty,
    label: str,
    reference: np.ndarray,
    converted: np.ndarray,
    limit: float,
) -> None:
    """Record the relative error of an output and assert the limit."""
    error = relative_error(reference, converted)
    record_property(f"{label} error", round(error, 5))
    assert error <= limit, f"{label}: relative error {error:.5f} > {limit}"


def locate_converted_model(output_dir: Path, platform: str) -> Path:
    """Locate the converted model a vendor inferer should load.

    Prefers the final artifact directly in ``output_dir`` -- running an
    intermediate DLC would test the un-quantized graph -- and falls back to a
    nested match for backends whose runnable model lives deeper (the rvc2/rvc3
    OpenVINO IR ``.xml``). A calibrated RVC3 blob is compiled from the INT8
    IR, so that IR wins over the FP16 IR next to it.
    """
    if platform == "rvc3" and (
        quantized := sorted(output_dir.rglob("optimized/*-int8.xml"))
    ):
        return quantized[0]
    pattern = _MODEL_GLOB[platform]
    matches = sorted(output_dir.glob(pattern)) or sorted(
        output_dir.rglob(pattern)
    )
    assert matches, (
        f"no converted model matching {pattern!r} under {output_dir}"
    )
    return matches[0]
