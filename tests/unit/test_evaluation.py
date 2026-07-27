"""Unit tests for the real-model evaluation test helpers."""

import numpy as np
import pytest

from modelconverter.utils.types import ResizeMethod
from tests.helpers.evaluation import (
    assert_quality,
    ordered_outputs,
    require_resize,
)


def test_ordered_outputs_uses_configured_order() -> None:
    first = np.array([1])
    second = np.array([2])

    ordered = ordered_outputs(
        {"second": second, "first": first}, ["first", "second"]
    )

    assert ordered == [first, second]


def test_ordered_outputs_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match=r"missing=.*second"):
        ordered_outputs({"first": np.array([1])}, ["first", "second"])


def test_require_resize_rejects_pad() -> None:
    require_resize(ResizeMethod.RESIZE)

    with pytest.raises(ValueError, match="resize_method=RESIZE"):
        require_resize(ResizeMethod.PAD)


def test_assert_quality_checks_source_floor_and_drop() -> None:
    assert_quality(
        {"AP": 0.40, "AP50": 0.60},
        {"AP": 0.38, "AP50": 0.59},
        floors={"AP": 0.39, "AP50": 0.59},
        max_drops={"AP": 0.03, "AP50": 0.02},
        case_id="test",
        platform="rvc4",
    )

    with pytest.raises(AssertionError, match="regressed"):
        assert_quality(
            {"AP": 0.40},
            {"AP": 0.30},
            floors={"AP": 0.39},
            max_drops={"AP": 0.03},
            case_id="test",
            platform="rvc4",
        )
