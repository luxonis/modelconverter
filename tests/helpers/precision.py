"""Numeric-fidelity helpers for the conversion precision tests.

A converted model counts as faithful if each output stays close -- by cosine
similarity -- to the original ONNX's output on the same image. Cosine similarity
tolerates the overall scale shifts quantization introduces while still catching
a model whose response has actually changed.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from modelconverter.utils.config import InputConfig
from modelconverter.utils.onnx_tools import onnx_attach_normalization_to_inputs

# The max IR version onnxruntime accepts varies by container (the Hailo 2024.10
# image caps at 8). Clamping the golden model keeps it loadable everywhere; the
# toy nets are opset 13, which only needs IR >= 7.
_MAX_ORT_IR_VERSION = 8

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
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def golden_reference_outputs(
    onnx_path: Path,
    input_configs: dict[str, InputConfig],
    work_dir: Path,
    value: float,
) -> dict[str, np.ndarray]:
    """Fp32 reference outputs the converted model should reproduce.

    Bakes the config's preprocessing into the ONNX with modelconverter's own
    ``onnx_attach_normalization_to_inputs`` -- the exact transform rvc2/rvc4
    apply before the backend step -- then runs it on a constant ``value`` image.
    Reusing that function avoids re-deriving the channel-reversal and
    mean/scale-reversal semantics by hand.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    golden = onnx_attach_normalization_to_inputs(
        Path(onnx_path), work_dir / "golden.onnx", input_configs
    )
    model = onnx.load(str(golden))
    model.ir_version = min(model.ir_version, _MAX_ORT_IR_VERSION)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feed = {
        i.name: np.full(i.shape, value, dtype=np.float32)
        for i in session.get_inputs()
    }
    names = [o.name for o in session.get_outputs()]
    return dict(
        zip(names, map(np.asarray, session.run(names, feed)), strict=True)
    )


def locate_converted_model(output_dir: Path, platform: str) -> Path:
    """The converted model a vendor inferer should load.

    Prefers the final artifact directly in ``output_dir`` -- running an
    intermediate DLC would test the un-quantized graph -- and falls back to a
    nested match for backends whose runnable model lives deeper (the rvc2/rvc3
    OpenVINO IR ``.xml``).
    """
    pattern = _MODEL_GLOB[platform]
    matches = sorted(output_dir.glob(pattern)) or sorted(
        output_dir.rglob(pattern)
    )
    assert matches, (
        f"no converted model matching {pattern!r} under {output_dir}"
    )
    return matches[0]
