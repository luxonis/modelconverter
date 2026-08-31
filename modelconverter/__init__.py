"""Model converter for Luxonis camera platforms.

``modelconverter`` converts a trained model into the format required by
the RVC2, RVC3, RVC4 or Hailo platform, passed either directly or
packaged in an NN Archive. ONNX is the format every platform accepts;
which other formats a platform takes is a property of its toolchain and
is documented by the corresponding sub-package of
`modelconverter.platforms`. Each conversion is executed inside a
per-backend Docker image that bundles the corresponding vendor
toolchain, so the host only needs Docker and this package.

This top-level package holds the version constants and registers any
externally provided put-file plugins at import time.
"""

from importlib.metadata import entry_points
from typing import Final

from luxonis_ml.utils import PUT_FILE_REGISTRY
from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.6.0"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)


def load_put_file_plugins() -> None:
    """Register any external put file plugins."""
    eps = entry_points()
    put_file_plugins = eps.select(group="put_file_plugins")
    for entry_point in put_file_plugins:  # pragma: no cover
        plugin_class = entry_point.load()
        PUT_FILE_REGISTRY.register(module=plugin_class)


load_put_file_plugins()
