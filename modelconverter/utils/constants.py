from pathlib import Path
from typing import Final

from luxonis_ml.utils.registry import Registry

SHARED_DIR: Final[Path] = Path("shared_with_container")
MISC_DIR: Final[Path] = SHARED_DIR / "misc"
CONFIGS_DIR: Final[Path] = SHARED_DIR / "configs"
OUTPUTS_DIR: Final[Path] = SHARED_DIR / "outputs"
CALIBRATION_DIR: Final[Path] = SHARED_DIR / "calibration_data"
MODELS_DIR: Final[Path] = SHARED_DIR / "models"
LOADERS = Registry(name="loaders")

__all__ = [
    "CALIBRATION_DIR",
    "CONFIGS_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "SHARED_DIR",
]
