"""Shared conversion options and SDK-specific skips.

modelconverter shell sets MODELCONVERTER_TOOL_VERSION for each SDK image.
"""

import os

from modelconverter.utils.types import Platform


def superblob_skip_reason() -> str | None:
    """Skip costly per-SHAVE compilation on OpenVINO 2021.4."""
    if os.getenv("MODELCONVERTER_TOOL_VERSION") == "2021.4.0":
        return "superblob compilation is too slow on OpenVINO 2021.4.0"
    return None


def platform_options(
    platform: Platform, *, superblob: bool = False
) -> tuple[str, ...]:
    """Use plain RVC2 blobs except where a test covers the default superblob."""
    if platform is Platform.RVC2 and (
        not superblob or superblob_skip_reason() is not None
    ):
        return ("rvc2.superblob", "False")
    return ()
