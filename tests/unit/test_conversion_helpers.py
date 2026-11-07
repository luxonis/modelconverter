"""Tests for conversion test helpers."""

import numpy as np
import pytest

from modelconverter.utils.types import Platform
from tests.helpers.platform_options import (
    platform_options,
    superblob_skip_reason,
)
from tests.helpers.precision import relative_error


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


def test_relative_error_rejects_different_shapes():
    with pytest.raises(AssertionError, match="output shape mismatch"):
        relative_error(np.zeros((1, 2)), np.zeros((2, 1)))


@pytest.mark.parametrize(
    ("version", "expected"),
    [("2022.3.0", ()), ("2021.4.0", ("rvc2.superblob", "False"))],
)
def test_platform_options_keeps_the_default_superblob(
    monkeypatch: pytest.MonkeyPatch, version: str, expected: tuple[str, ...]
):
    monkeypatch.setenv("MODELCONVERTER_TOOL_VERSION", version)

    assert platform_options(Platform.RVC2, superblob=True) == expected


@pytest.mark.parametrize(
    ("version", "skipped"),
    [("2021.4.0", True), ("2022.3.0", False)],
)
def test_superblob_skip_reason(
    monkeypatch: pytest.MonkeyPatch, version: str, *, skipped: bool
):
    monkeypatch.setenv("MODELCONVERTER_TOOL_VERSION", version)

    assert (superblob_skip_reason() is not None) is skipped
