"""Conversion options and skips forced by the target's tool version.

``MODELCONVERTER_TARGET_VERSION`` is injected by ``modelconverter shell``, so a
test can tell which vendor toolchain it is running against.
"""

import os

from modelconverter.utils.types import Target


def superblob_skip_reason() -> str | None:
    """Why superblob is not exercised on this tool version, if it is not.

    A superblob compiles one patch per shave count, and on OpenVINO 2021.4.0
    that is slow enough to dominate the CI job -- so the RVC2 tests build a
    plain blob there. It is a runtime budget, not a capability: superblob works
    on 2021.4.0, it is just too expensive to compile on every run.
    """
    if os.getenv("MODELCONVERTER_TARGET_VERSION") == "2021.4.0":
        return "superblob compilation is too slow on OpenVINO 2021.4.0"
    return None


def target_options(target: Target) -> tuple[str, ...]:
    """Conversion overrides this target's tool version requires.

    Every conversion test threads these into ``convert`` (and into any
    ``get_configs`` call that has to agree with it), so a tool version that
    needs different settings is handled in one place rather than per test.
    """
    if target is Target.RVC2 and superblob_skip_reason() is not None:
        return ("rvc2.superblob", "False")
    return ()
