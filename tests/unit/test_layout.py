"""Unit tests for ``modelconverter.utils.layout``.

The two functions are pure and total over shapes, so most of the
coverage here is property-based: the hand-written cases below pin the
handful of *specific* layouts the rest of the converter relies on
(``NCHW``, ``NHWC``, the alphabet fallback), and the properties assert
the invariants that must hold for every shape.
"""

import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from modelconverter.utils.layout import guess_new_layout, make_default_layout
from tests.helpers.strategies import shape_permutations, shapes


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ([1, 3, 256, 256], "NCHW"),
        ([1, 256, 256, 3], "NHWC"),
        # Leading 1 -> "N", remaining three dims are not min-channel
        # patterns, so the alphabet fallback (starting at "C") kicks in.
        ([1, 19, 7, 8], "NCDE"),
    ],
)
def test_recognised_layouts(shape: list[int], expected: str):
    assert make_default_layout(shape) == expected


def test_no_leading_one():
    layout = make_default_layout([3, 4, 5])
    assert len(layout) == 3
    assert "N" not in layout


def test_letter_collision_loop():
    # A shape long enough that the alphabet counter reaches the
    # already-used "N" (at i==11) forces the skip branch of the loop.
    shape = [1, *range(2, 15)]  # 14 dims
    layout = make_default_layout(shape)
    assert len(set(layout)) == len(shape)
    assert "N" in layout


def test_transpose():
    assert (
        guess_new_layout("NCHW", [1, 3, 256, 256], [1, 256, 256, 3]) == "NHWC"
    )


def test_duplicate_dims_preserve_order():
    assert guess_new_layout("NCHW", [1, 3, 3, 4], [1, 3, 4, 3]) == "NCWH"


def test_length_mismatch():
    with pytest.raises(ValueError, match="same as the old one"):
        guess_new_layout("NCHW", [1, 3, 256, 256], [1, 256, 256])


def test_element_mismatch():
    with pytest.raises(ValueError, match="same elements"):
        guess_new_layout("NCHW", [1, 3, 256, 256], [1, 3, 256, 128])


@given(shape=shapes())
def test_every_dimension_gets_a_distinct_letter(shape: list[int]):
    layout = make_default_layout(shape)
    assert len(layout) == len(shape)
    assert len(set(layout)) == len(shape)
    assert set(layout) <= set(string.ascii_uppercase)


@given(rest=shapes(max_rank=15))
def test_leading_one_is_the_batch_dimension(rest: list[int]):
    assert make_default_layout([1, *rest]).startswith("N")


@given(shape_and_permutation=shape_permutations())
def test_guessed_layout_follows_its_dimensions(
    shape_and_permutation: tuple[list[int], list[int]],
):
    """Every label ends up on a dimension of the same size.

    This is the whole contract of ``guess_new_layout``: relabelling a
    permuted shape may not invent, drop or resize a label.
    """
    shape, new_shape = shape_and_permutation
    layout = make_default_layout(shape)

    new_layout = guess_new_layout(layout, shape, new_shape)

    assert new_layout is not None
    assert sorted(new_layout) == sorted(layout)
    assert sorted(zip(new_layout, new_shape, strict=True)) == sorted(
        zip(layout, shape, strict=True)
    )


@given(shape=shapes())
def test_identity_permutation_keeps_the_layout(shape: list[int]):
    layout = make_default_layout(shape)
    assert guess_new_layout(layout, shape, shape) == layout


@given(shape=shapes(), extra=st.integers(min_value=1, max_value=64))
def test_extra_dimension_is_rejected(shape: list[int], extra: int):
    layout = make_default_layout(shape)
    with pytest.raises(ValueError, match="same as the old one"):
        guess_new_layout(layout, shape, [*shape, extra])
