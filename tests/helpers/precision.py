"""Numeric-fidelity helpers for the conversion precision tests.

A converted model is judged faithful if each of its outputs stays close --
by cosine similarity -- to the original ONNX model's output on the same
image. Cosine similarity (rather than an absolute diff) is robust to the
overall scale shifts quantization introduces while still catching a model
whose response has actually changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# The converted model file to feed the vendor inferer, per platform.
_MODEL_GLOB = {
    "rvc2": "*.xml",
    "rvc3": "*.xml",
    "rvc4": "*.dlc",
    "hailo": "*.har",
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays, compared as flat vectors."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(
            f"cannot compare outputs of shape {a.shape} and {b.shape}"
        )
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.dot(x, y) / denom)


def locate_converted_model(output_dir: Path, platform: str) -> Path:
    """Find the converted model file a vendor inferer should load."""
    pattern = _MODEL_GLOB[platform]
    matches = sorted(output_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no converted model matching {pattern!r} under {output_dir}"
        )
    return matches[0]
