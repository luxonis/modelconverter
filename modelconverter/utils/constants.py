from pathlib import Path
from typing import Final

from luxonis_ml.utils.registry import Registry

SHARED_DIR: Final[Path] = Path("shared_with_container")
MISC_DIR: Final[Path] = SHARED_DIR / "misc"
CONFIGS_DIR: Final[Path] = SHARED_DIR / "configs"
OUTPUTS_DIR: Final[Path] = SHARED_DIR / "outputs"
CALIBRATION_DIR: Final[Path] = SHARED_DIR / "calibration_data"
MODELS_DIR: Final[Path] = SHARED_DIR / "models"
RUNTIME_HOME_SUFFIX: Final[str] = "runtime-home"
RUNTIME_CACHE_SUFFIX: Final[str] = "runtime-cache"
LOADERS = Registry(name="loaders")

__all__ = [
    "CALIBRATION_DIR",
    "CONFIGS_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "RUNTIME_CACHE_SUFFIX",
    "RUNTIME_HOME_SUFFIX",
    "SHARED_DIR",
]
