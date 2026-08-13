import pytest

from modelconverter.utils.types import Platform
from tests.helpers.platform_options import (
    platform_options,
    superblob_skip_reason,
)


@pytest.mark.parametrize(
    ("platform", "version", "expected"),
    [
        (Platform.RVC2, "2021.4.0", ("rvc2.superblob", "False")),
        (Platform.RVC2, "2022.3.0", ()),
        (Platform.RVC3, "2021.4.0", ()),
        (Platform.RVC4, "2.41.0", ()),
        (Platform.HAILO, "2025.04", ()),
    ],
)
def test_platform_options(
    monkeypatch: pytest.MonkeyPatch,
    platform: Platform,
    version: str,
    expected: tuple[str, ...],
):
    monkeypatch.setenv("MODELCONVERTER_TOOL_VERSION", version)

    assert platform_options(platform) == expected


@pytest.mark.parametrize(
    ("version", "skipped"),
    [("2021.4.0", True), ("2022.3.0", False)],
)
def test_superblob_skip_reason(
    monkeypatch: pytest.MonkeyPatch, version: str, skipped: bool
):
    monkeypatch.setenv("MODELCONVERTER_TOOL_VERSION", version)

    assert (superblob_skip_reason() is not None) is skipped


def test_no_target_version_keeps_superblob(
    monkeypatch: pytest.MonkeyPatch,
):
    """Outside a conversion container nothing is forced off.

    Host-side runs have no ``MODELCONVERTER_TOOL_VERSION``, and defaulting to
    "skip superblob" there would silently weaken every RVC2 conversion.
    """
    monkeypatch.delenv("MODELCONVERTER_TOOL_VERSION", raising=False)

    assert superblob_skip_reason() is None
    assert platform_options(Platform.RVC2) == ()
