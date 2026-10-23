"""Tests for the per-target conversion option helpers.

Covers the target-specific CLI options and the conditions under which the
superblob option is skipped.
"""

import pytest

from modelconverter.utils.types import Platform
from tests.helpers.platform_options import (
    platform_options,
    superblob_skip_reason,
)


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        (Platform.RVC2, ("rvc2.superblob", "False")),
        (Platform.RVC3, ()),
        (Platform.RVC4, ()),
        (Platform.HAILO, ()),
    ],
)
def test_platform_options(platform: Platform, expected: tuple[str, ...]):
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
