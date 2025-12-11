from importlib.metadata import entry_points
from typing import Final

from luxonis_ml.utils import PUT_FILE_REGISTRY
from pydantic_extra_types.semantic_version import SemanticVersion

__version__: Final[str] = "0.5.1"
__semver__: Final[SemanticVersion] = SemanticVersion.parse(__version__)


def load_put_file_plugins() -> None:
    """Registers any external put file plugins."""
    eps = entry_points()
    put_file_plugins = eps.select(group="put_file_plugins")
    for entry_point in put_file_plugins:
        plugin_class = entry_point.load()
        PUT_FILE_REGISTRY.register_module(module=plugin_class)


load_put_file_plugins()
