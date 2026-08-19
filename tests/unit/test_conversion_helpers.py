"""Tests for the per-target conversion option helpers.

Covers the target-specific CLI options and the conditions under which the
superblob option is skipped.
"""

import pytest

from modelconverter.utils.types import Target
from tests.helpers.target_options import (
    superblob_skip_reason,
    target_options,
)


@pytest.mark.parametrize(
    ("target", "version", "expected"),
    [
        (Target.RVC2, "2021.4.0", ("rvc2.superblob", "False")),
        (Target.RVC2, "2022.3.0", ()),
        (Target.RVC3, "2021.4.0", ()),
        (Target.RVC4, "2.41.0", ()),
        (Target.HAILO, "2025.04", ()),
    ],
)
def test_target_options(
    monkeypatch: pytest.MonkeyPatch,
    target: Target,
    version: str,
    expected: tuple[str, ...],
):
    monkeypatch.setenv("MODELCONVERTER_TARGET_VERSION", version)

    assert target_options(target) == expected


@pytest.mark.parametrize(
    ("version", "skipped"),
    [("2021.4.0", True), ("2022.3.0", False)],
)
def test_superblob_skip_reason(
    monkeypatch: pytest.MonkeyPatch, version: str, skipped: bool
):
    monkeypatch.setenv("MODELCONVERTER_TARGET_VERSION", version)

    assert (superblob_skip_reason() is not None) is skipped


def test_no_target_version_keeps_superblob(
    monkeypatch: pytest.MonkeyPatch,
):
    """Outside a conversion container nothing is forced off.

    Host-side runs have no ``MODELCONVERTER_TARGET_VERSION``, and defaulting to
    "skip superblob" there would silently weaken every RVC2 conversion.
    """
    monkeypatch.delenv("MODELCONVERTER_TARGET_VERSION", raising=False)

    assert superblob_skip_reason() is None
    assert target_options(Target.RVC2) == ()
