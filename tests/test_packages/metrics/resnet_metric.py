from pathlib import Path
from typing import Dict, Union

import numpy as np

from modelconverter.utils import read_image
from modelconverter.utils.types import Encoding, ResizeMethod

from ..onnx_inferer import ONNXInferer
from .base_metric import Metric


class ResnetMetric(Metric):
    def __init__(self):
        self.hits = 0
        self.misses = 0

    def update(self, output: np.ndarray, label: int) -> None:
        pred = np.argmax(output)
        if label == pred:
            self.hits += 1
        else:
            self.misses += 1

    def get_result(self) -> Dict[str, float]:
        return {"accuracy": self.hits / (self.hits + self.misses)}

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0

    @staticmethod
    def eval_onnx(onnx_path: str, dataset_path: Union[Path, str]):
        dataset_path = Path(dataset_path)
        onnx_inferer = ONNXInferer(onnx_path)
        metric = ResnetMetric()
        for label in dataset_path.iterdir():
            for img_path in label.iterdir():
                img = read_image(
                    img_path,
                    shape=[1, 3, 256, 256],
                    encoding=Encoding.RGB,
                    resize_method=ResizeMethod.RESIZE,
                    transpose=True,
                ).astype(np.float32)[np.newaxis, ...]
                img -= np.array([123.675, 116.28, 103.53]).reshape(1, 3, 1, 1)
                img /= np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

                output = onnx_inferer.infer({"input.1": img})["191"]
                metric.update(output, int(label.stem))
        return metric.get_result()
