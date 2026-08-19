"""Helpers backing the ``modelconverter`` command-line interface.

Re-exports everything from ``modelconverter.cli.utils``: turning what
was passed on the command line (a config file, an NN Archive, a bare
model file and ``key value`` overrides) into a parsed configuration,
preparing the shared directories and resolving the output directory.
"""

from .utils import *
