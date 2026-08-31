"""Base class of the analysis result visualizers.

The analyzers write their per-layer measurements as CSV files; the
per-platform subclasses of `Visualizer` turn those files into plots.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from modelconverter.utils.constants import OUTPUTS_DIR


class Visualizer(ABC):
    """Base class for visualizing the results of a model analysis.

    Attributes:
        _dir_path: Directory the analysis results are read from.

    """

    def __init__(self, dir_path: str | None = None) -> None:
        """Initialize the visualizer.

        Args:
            dir_path: Directory containing the analysis results.
                Defaults to the ``analysis`` subdirectory of the
                modelconverter output directory.

        """
        self._dir_path = Path(dir_path or OUTPUTS_DIR / "analysis")

    @abstractmethod
    def visualize(self) -> None:
        """Visualize the analysis results found in ``_dir_path``."""
