"""Platform parametrization for the conversion tests.

Every conversion case carries its platform's marker, so ``pytest -m rvc2``
selects only the RVC2 conversions.
"""

import pytest

ALL_PLATFORMS = ("rvc2", "rvc3", "rvc4", "hailo")


def platform_marks(platform: str, xfail: str | None = None) -> list:
    """The platform marker, plus a strict xfail when ``xfail`` gives a reason.

    A failing conversion exits rather than raising, hence ``raises=SystemExit``.
    """
    marks = [getattr(pytest.mark, platform)]
    if xfail is not None:
        marks.append(
            pytest.mark.xfail(reason=xfail, strict=True, raises=SystemExit)
        )
    return marks


def platform_params(
    platforms: tuple[str, ...] = ALL_PLATFORMS,
    xfails: dict[str, str] | None = None,
) -> list:
    """One ``pytest.param`` per platform, for a ``"platform"`` parametrization."""
    return [
        pytest.param(
            platform,
            marks=platform_marks(platform, (xfails or {}).get(platform)),
            id=platform,
        )
        for platform in platforms
    ]
