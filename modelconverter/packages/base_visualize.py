from abc import ABC, abstractmethod

from modelconverter.utils import constants


class Visualizer(ABC):
    def __init__(self, dir_path: str = "") -> None:
        self.dir_path: str = (
            dir_path
            if dir_path != ""
            else f"{str(constants.OUTPUTS_DIR)}/analysis"
        )

    @abstractmethod
    def visualize(self) -> None:
        pass
