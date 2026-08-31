"""Per-platform conversion, inference and benchmarking packages.

Every supported platform -- RVC2, RVC3, RVC4 and Hailo -- has its own
sub-package implementing the abstract bases defined here. All four
provide an exporter and an inferer, every platform but Hailo also
provides a benchmark, and only RVC4 provides an analyzer and a
visualizer. Each implementation drives the vendor toolchain and
therefore only runs inside that platform's Docker image. The getters
re-exported from this package pick the implementation matching the
requested platform.
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
