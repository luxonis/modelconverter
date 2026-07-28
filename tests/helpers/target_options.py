"""Shared target-specific conversion test options."""

import os

from modelconverter.utils.types import Target


def target_options(target: Target) -> tuple[str, ...]:
    """Return options required by a target tool-version combination."""
    if (
        target is Target.RVC2
        and os.getenv("MODELCONVERTER_TARGET_VERSION") == "2021.4.0"
    ):
        return ("rvc2.superblob", "False")
    return ()
