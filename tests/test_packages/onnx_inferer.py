from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


class ONNXInferer:
    def __init__(self, onnx_path: str | Path) -> None:
        self.onnx_path = Path(onnx_path)
        self.model = onnx.load(str(self.onnx_path))
        self.output_names = [output.name for output in self.model.graph.output]
        self.session = ort.InferenceSession(str(self.onnx_path))

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self.session.run(self.output_names, inputs)
        return dict(zip(self.output_names, outputs, strict=True))
