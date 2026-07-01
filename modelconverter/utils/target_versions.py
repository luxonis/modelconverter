from typing import Literal


def get_default_target_version(
    target: Literal["rvc2", "rvc3", "rvc4", "hailo"],
) -> str:
    return {
        "rvc2": "2022.3.0",
        "rvc3": "2022.3.0",
        "rvc4": "2.41.0",
        "hailo": "2025.04",
    }[target]
