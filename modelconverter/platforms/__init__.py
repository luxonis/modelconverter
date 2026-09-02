"""Per-platform conversion, inference and benchmarking packages.

Every supported platform -- RVC2, RVC3, RVC4 and Hailo -- has its own
sub-package implementing the abstract bases defined here. All four
provide an exporter and an inferer, every platform but Hailo also
provides a benchmark, and only RVC4 provides an analyzer and a
visualizer. The exporters and inferers drive the vendor toolchains
and run inside the platform's Docker image. The benchmarks, the
analyzer and the visualizer run on the host. The getters re-exported
from this package pick the implementation matching the requested
platform.
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
