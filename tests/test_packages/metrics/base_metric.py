from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Metric(ABC):
    @abstractmethod
    def update(self, output: list[np.ndarray], label: np.ndarray): ...

    @abstractmethod
    def get_result(self) -> dict[str, float]: ...

    @abstractmethod
    def reset(self): ...

    @staticmethod
    @abstractmethod
    def eval_onnx(
        onnx_path: Path | str, dataset_path: Path | str
    ) -> dict[str, float]: ...
