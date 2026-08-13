"""Per-target conversion, inference and benchmarking packages.

Every supported target -- RVC2, RVC3, RVC4 and Hailo -- has its own
sub-package implementing the abstract bases defined here. All four
provide an exporter and an inferer, every target but Hailo also
provides a benchmark, and only RVC4 provides an analyzer and a
visualizer. Each implementation drives the vendor toolchain and
therefore only runs inside that target's Docker image. The getters
re-exported from this package pick the implementation matching the
requested target.
"""

from .getters import (
    get_analyzer,
    get_benchmark,
    get_exporter,
    get_inferer,
    get_visualizer,
)

__all__ = [
    "get_analyzer",
    "get_benchmark",
    "get_exporter",
    "get_inferer",
    "get_visualizer",
]
