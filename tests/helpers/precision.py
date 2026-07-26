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
import onnxruntime as ort

from modelconverter.utils.config import InputConfig
from modelconverter.utils.onnx_tools import onnx_attach_normalization_to_inputs

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


def golden_reference_outputs(
    onnx_path: Path,
    input_configs: dict[str, InputConfig],
    work_dir: Path,
    value: float,
) -> dict[str, np.ndarray]:
    """Fp32 reference outputs the converted model should reproduce.

    Bakes the config's preprocessing into the ONNX with modelconverter's own
    ``onnx_attach_normalization_to_inputs`` -- the exact transform rvc2/rvc4
    apply before the backend step -- then runs it in onnxruntime on a
    constant ``value`` image (fed to every input). Using modelconverter's own
    preprocessing avoids re-deriving the channel-reversal / mean-scale-reversal
    semantics by hand.
    """
    golden = onnx_attach_normalization_to_inputs(
        Path(onnx_path), work_dir / "golden.onnx", input_configs
    )
    session = ort.InferenceSession(
        str(golden), providers=["CPUExecutionProvider"]
    )
    feed = {
        i.name: np.full(i.shape, value, dtype=np.float32)
        for i in session.get_inputs()
    }
    names = [o.name for o in session.get_outputs()]
    outputs = session.run(names, feed)
    return {
        name: np.asarray(out) for name, out in zip(names, outputs, strict=True)
    }


def locate_converted_model(output_dir: Path, platform: str) -> Path:
    """Find the converted model file a vendor inferer should load.

    Prefer the *final* artifact directly in ``output_dir`` (e.g. the final
    quantized rvc4 ``.dlc``) over the intermediate copies under
    ``intermediate_outputs/`` -- running an intermediate DLC would test the
    un-quantized graph. Fall back to a nested match for backends whose
    runnable model lives deeper (the rvc2/rvc3 OpenVINO IR ``.xml``).
    """
    pattern = _MODEL_GLOB[platform]
    matches = sorted(output_dir.glob(pattern)) or sorted(
        output_dir.rglob(pattern)
    )
    if not matches:
        raise FileNotFoundError(
            f"no converted model matching {pattern!r} under {output_dir}"
        )
    return matches[0]
