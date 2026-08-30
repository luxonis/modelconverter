"""Host-side unit tests for ``modelconverter.utils.tool_versions``."""

from typing import Literal

import pytest

from modelconverter.utils.tool_versions import get_default_tool_version


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("rvc2", "2022.3.0"),
        ("rvc3", "2022.3.0"),
        ("rvc4", "2.41.0"),
        ("hailo", "2025.04"),
    ],
)
def test_get_default_tool_version(
    platform: Literal["rvc2", "rvc3", "rvc4", "hailo"], expected: str
):
    assert get_default_tool_version(platform) == expected
