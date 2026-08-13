import os
from pathlib import Path, PurePosixPath
from typing import Final

from luxonis_ml.utils.registry import Registry


def in_docker() -> bool:
    """Whether this process runs inside a modelconverter container."""
    return "IN_DOCKER" in os.environ


def get_cache_dir() -> Path:
    """Return the hidden, auto-managed cache directory used for staged
    inputs and remote downloads.

    Respects ``XDG_CACHE_HOME`` and defaults to
    ``~/.cache/modelconverter``.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "modelconverter"


# Inside the container the cache is bind-mounted at a fixed location and the
# outputs are written to a dedicated mount that maps back to `./output` on the
# host. Outside the container (native runs, host-only commands, tests) we use
# the on-disk cache directory and a local `./output` directory.
_IN_DOCKER: Final[bool] = in_docker()

# A container-side location: pure so that paths built from it keep their
# forward slashes even when the launcher runs on a Windows host.
CONTAINER_SHARED_DIR: Final[PurePosixPath] = PurePosixPath(
    "/app/shared_with_container"
)

SHARED_DIR: Final[Path] = (
    Path(CONTAINER_SHARED_DIR) if _IN_DOCKER else get_cache_dir()
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

# Written into a directory modelconverter produced itself. `OUTPUTS_DIR` is the
# host's `./output`, so a rerun may only clear a directory it made; anything
# else the user keeps there is not ours to delete. The two kinds are separate
# on purpose: an inference rerun must not wipe a conversion's output either.
CONVERSION_MARKER: Final[str] = ".modelconverter_output"
INFERENCE_MARKER: Final[str] = ".modelconverter_inference"

LOADERS = Registry(name="loaders")

__all__ = [
    "CALIBRATION_DIR",
    "CONFIGS_DIR",
    "CONTAINER_SHARED_DIR",
    "CONVERSION_MARKER",
    "INFERENCE_MARKER",
    "MISC_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "SHARED_DIR",
    "get_cache_dir",
    "in_docker",
]
