"""Conversion options shared by the conversion tests, and the skips that
the platform's tool version forces.

``MODELCONVERTER_TOOL_VERSION`` is injected by ``modelconverter shell``, so a
test can tell which vendor toolchain it is running against.
"""

import os

from modelconverter.utils.types import Platform


def superblob_skip_reason() -> str | None:
    """Why superblob is not exercised on this tool version, if it is not.

    A superblob compiles one patch per shave count, and on OpenVINO 2021.4.0
    that is slow enough to dominate the CI job -- so the dedicated superblob
    case is skipped there. It is a runtime budget, not a capability: superblob
    works on 2021.4.0, it is just too expensive to compile on every run.
    """
    if os.getenv("MODELCONVERTER_TOOL_VERSION") == "2021.4.0":
        return "superblob compilation is too slow on OpenVINO 2021.4.0"
    return None


def platform_options(platform: Platform) -> tuple[str, ...]:
    """Conversion overrides that every conversion test applies.

    Every conversion test threads these into ``convert`` (and into any
    ``get_configs`` call that has to agree with it), so the settings are
    handled in one place rather than per test.

    RVC2 builds a plain blob. A superblob compiles one blob per SHAVE
    count, and only the superblob case of ``test_toy_integration`` checks
    one.
    """
    if platform is Platform.RVC2:
        return ("rvc2.superblob", "False")
    return ()
