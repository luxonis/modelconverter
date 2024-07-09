from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union


class Metric(ABC):
    @abstractmethod
    def update(self, output: Any, label: Any):
        pass

    @abstractmethod
    def get_result(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @staticmethod
    @abstractmethod
    def eval_onnx(
        onnx_path: Union[Path, str], dataset_path: Union[Path, str]
    ) -> Dict[str, float]:
        pass
