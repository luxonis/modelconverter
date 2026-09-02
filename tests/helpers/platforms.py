"""Platform parametrization for the conversion tests.

Every conversion case carries its platform's marker, so ``pytest -m rvc2``
selects only the RVC2 conversions.
"""

import pytest

ALL_PLATFORMS = ("rvc2", "rvc3", "rvc4", "hailo")


def platform_marks(
    platform: str, xfail: str | None = None, skip: str | None = None
) -> list:
    """Build the platform marker, plus an xfail or a skip when given a reason.

    A failing conversion exits rather than raising, hence ``raises=SystemExit``
    on the xfail. ``skip`` is for cases the current tool version will not run --
    unsupported, or too slow to be worth it -- which is not the same as a case
    that is expected to fail.
    """
    marks = [getattr(pytest.mark, platform)]
    if xfail is not None:
        marks.append(
            pytest.mark.xfail(reason=xfail, strict=True, raises=SystemExit)
        )
    if skip is not None:
        marks.append(pytest.mark.skip(reason=skip))
    return marks


def platform_params(
    platforms: tuple[str, ...] = ALL_PLATFORMS,
    xfails: dict[str, str] | None = None,
) -> list:
    """One ``pytest.param`` per platform name, for a ``"platform_name"``
    parametrization.
    """
    return [
        pytest.param(
            platform,
            marks=platform_marks(platform, (xfails or {}).get(platform)),
            id=platform,
        )
        for platform in platforms
    ]
