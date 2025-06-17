from importlib.metadata import entry_points

from luxonis_ml.utils import PUT_FILE_REGISTRY

from .hub import convert

__version__ = "0.4.1"

__all__ = ["convert"]


def load_put_file_plugins() -> None:
    """Registers any external put file plugins."""
    eps = entry_points()
    put_file_plugins = eps.select(group="put_file_plugins")
    for entry_point in put_file_plugins:
        plugin_class = entry_point.load()
        PUT_FILE_REGISTRY.register_module(module=plugin_class)


load_put_file_plugins()
