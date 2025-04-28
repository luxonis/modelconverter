from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Metric(ABC):
    @abstractmethod
    def update(self, output: Any, label: Any):
        pass

    @abstractmethod
    def get_result(self) -> dict[str, float]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @staticmethod
    @abstractmethod
    def eval_onnx(
        onnx_path: Path | str, dataset_path: Path | str
    ) -> dict[str, float]:
        pass
