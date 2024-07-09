from pathlib import Path
from typing import Dict, Union

import numpy as np

from modelconverter.utils import read_image
from modelconverter.utils.types import Encoding, ResizeMethod

from ..onnx_inferer import ONNXInferer
from .base_metric import Metric


class MNISTMetric(Metric):
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
    def eval_onnx(onnx_path: Union[Path, str], dataset_path: Union[Path, str]):
        dataset_path = Path(dataset_path)
        onnx_path = Path(onnx_path)
        onnx_inferer = ONNXInferer(onnx_path)
        metric = MNISTMetric()
        for img_path in dataset_path.iterdir():
            img = read_image(
                img_path,
                shape=[1, 1, 28, 28],
                encoding=Encoding.GRAY,
                resize_method=ResizeMethod.RESIZE,
                transpose=True,
            ).astype(np.float32)[np.newaxis, ...]
            img /= 255.0
            label = int(img_path.stem.split("_")[-1])
            output = onnx_inferer.infer({"input": img})["output"]
            metric.update(output, label)
        return metric.get_result()
