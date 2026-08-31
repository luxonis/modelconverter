"""Helpers backing the ``modelconverter`` command-line interface.

Re-exports the helpers from ``modelconverter.cli.utils``: turning what
was passed on the command line (a config file, an NN Archive, a bare
model file and ``key value`` overrides) into a parsed configuration,
preparing the shared directories and resolving the output directory.
"""

from .utils import (
    extract_preprocessing,
    get_configs,
    get_output_dir_name,
    init_dirs,
    resolve_output_dir,
)

__all__ = [
    "extract_preprocessing",
    "get_configs",
    "get_output_dir_name",
    "init_dirs",
    "resolve_output_dir",
]
