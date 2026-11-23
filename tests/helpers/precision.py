"""Precision metrics for converted model outputs.

Relative error catches scale mistakes that cosine similarity misses.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

_MODEL_GLOB = {
    "rvc2": "*.xml",
    "rvc3": "*.xml",
    "rvc4": "*.dlc",
    "hailo": "*.har",
}

RecordProperty = Callable[[str, object], None]


def cosine_similarity(
    a: npt.NDArray[np.generic], b: npt.NDArray[np.generic]
) -> float:
    """Cosine similarity between two arrays, compared as flat vectors."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0 or y_norm == 0:
        return 1.0 if x_norm == y_norm else 0.0
    return float(np.dot(x, y) / (x_norm * y_norm))


def relative_error(
    reference: npt.NDArray[np.generic],
    converted: npt.NDArray[np.generic],
) -> float:
    """L2 norm of the difference, relative to the L2 norm of the reference."""
    assert reference.shape == converted.shape, (
        f"output shape mismatch: {reference.shape} vs {converted.shape}"
    )
    x = np.asarray(reference, dtype=np.float64).ravel()
    y = np.asarray(converted, dtype=np.float64).ravel()
    reference_norm = np.linalg.norm(x)
    difference_norm = np.linalg.norm(x - y)
    if reference_norm == 0:
        return 0.0 if difference_norm == 0 else float("inf")
    return float(difference_norm / reference_norm)


def check_relative_error(
    record_property: RecordProperty,
    label: str,
    reference: npt.NDArray[np.generic],
    converted: npt.NDArray[np.generic],
    limit: float,
) -> None:
    error = relative_error(reference, converted)
    record_property(f"{label} error", round(error, 5))
    assert error <= limit, f"{label}: relative error {error:.5f} > {limit}"


def locate_converted_model(output_dir: Path, platform: str) -> Path:
    """Locate the converted model a vendor inferer should load.

    Prefers the final artifact directly in ``output_dir`` -- running an
    intermediate DLC would test the un-quantized graph -- and falls back to a
    nested match for backends whose runnable model lives deeper. A calibrated
    RVC3 blob is compiled from the INT8 IR, so that IR wins over the FP16 IR
    next to it.

    Returns:
        The first RVC3 INT8 IR when one exists. Otherwise, the first direct
        artifact match or the first nested match, in sorted order.
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
