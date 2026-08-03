import os
from pathlib import Path
from typing import Final

from luxonis_ml.utils.registry import Registry


def in_docker() -> bool:
    """Whether this process runs inside a modelconverter container."""
    return "IN_DOCKER" in os.environ


def get_cache_dir() -> Path:
    """Returns the hidden, auto-managed cache directory used for staged
    inputs and remote downloads.

    Respects C{XDG_CACHE_HOME} and defaults to
    C{~/.cache/modelconverter}.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "modelconverter"


# Inside the container the cache is bind-mounted at a fixed location and the
# outputs are written to a dedicated mount that maps back to `./output` on the
# host. Outside the container (native runs, host-only commands, tests) we use
# the on-disk cache directory and a local `./output` directory.
_IN_DOCKER: Final[bool] = in_docker()

CONTAINER_SHARED_DIR: Final[Path] = Path("/app/shared_with_container")

SHARED_DIR: Final[Path] = (
    CONTAINER_SHARED_DIR if _IN_DOCKER else get_cache_dir()
)
OUTPUTS_DIR: Final[Path] = (
    Path("/app/output") if _IN_DOCKER else Path.cwd() / "output"
)

# Destinations for remote downloads and scratch data. These always live inside
# the cache directory so they are covered by `cache clean`.
MISC_DIR: Final[Path] = SHARED_DIR / "misc"
CONFIGS_DIR: Final[Path] = SHARED_DIR / "configs"
CALIBRATION_DIR: Final[Path] = SHARED_DIR / "calibration_data"
MODELS_DIR: Final[Path] = SHARED_DIR / "models"

# Directory holding staged copies of user-provided inputs, keyed by content
# hash for de-duplication across runs.
INPUTS_DIR: Final[Path] = SHARED_DIR / "inputs"

LOADERS = Registry(name="loaders")

__all__ = [
    "CALIBRATION_DIR",
    "CONFIGS_DIR",
    "CONTAINER_SHARED_DIR",
    "INPUTS_DIR",
    "MISC_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "SHARED_DIR",
    "get_cache_dir",
    "in_docker",
]
