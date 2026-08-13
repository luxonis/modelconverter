"""Default conversion tool versions of the supported targets.

Each target is converted by a vendor toolchain that ships in its own
Docker image, and several versions of it may be available. This module
holds the version used when the user does not pass ``--tool-version``.
"""

from typing import Literal


def get_default_target_version(
    target: Literal["rvc2", "rvc3", "rvc4", "hailo"],
) -> str:
    """Return the default conversion tool version for a target.

    Args:
        target: Name of the conversion target.

    Returns:
        Version string of the toolchain used for ``target`` by default.

    """
    return {
        "rvc2": "2022.3.0",
        "rvc3": "2022.3.0",
        "rvc4": "2.41.0",
        "hailo": "2025.04",
    }[target]
