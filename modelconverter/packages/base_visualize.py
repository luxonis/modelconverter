from abc import ABC, abstractmethod
from pathlib import Path

from modelconverter.utils import constants


class Visualizer(ABC):
    def __init__(self, dir_path: str = "") -> None:
        self.dir_path = Path(
            dir_path
            if dir_path != ""
            else f"{constants.OUTPUTS_DIR!s}/analysis"
        )

    @abstractmethod
    def visualize(self) -> None:
        pass
