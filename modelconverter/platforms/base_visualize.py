from abc import ABC, abstractmethod
from pathlib import Path

from modelconverter.utils.constants import OUTPUTS_DIR


class Visualizer(ABC):
    def __init__(self, dir_path: str | None = None) -> None:
        self.dir_path = Path(dir_path or OUTPUTS_DIR / "analysis")

    @abstractmethod
    def visualize(self) -> None: ...
