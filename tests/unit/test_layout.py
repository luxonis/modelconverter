import pytest

from modelconverter.utils.layout import guess_new_layout, make_default_layout


class TestMakeDefaultLayout:
    def test_nchw(self):
        assert make_default_layout([1, 3, 256, 256]) == "NCHW"

    def test_nhwc(self):
        assert make_default_layout([1, 256, 256, 3]) == "NHWC"

    def test_alphabet_fallback(self):
        # Leading 1 -> "N", remaining three dims are not min-channel
        # patterns, so the alphabet fallback (starting at "C") kicks in.
        assert make_default_layout([1, 19, 7, 8]) == "NCDE"

    def test_no_leading_one(self):
        layout = make_default_layout([3, 4, 5])
        assert len(layout) == 3
        assert "N" not in layout

    def test_letter_collision_loop(self):
        # A shape long enough that the alphabet counter reaches the
        # already-used "N" (at i==11) forces the skip branch of the loop.
        shape = [1, *range(2, 15)]  # 14 dims
        layout = make_default_layout(shape)
        assert len(layout) == len(shape)
        assert len(set(layout)) == len(shape)  # all unique
        assert "N" in layout


class TestGuessNewLayout:
    def test_transpose(self):
        assert (
            guess_new_layout("NCHW", [1, 3, 256, 256], [1, 256, 256, 3])
            == "NHWC"
        )

    def test_duplicate_dims_preserve_order(self):
        assert (
            guess_new_layout("NCHW", [1, 3, 3, 4], [1, 3, 4, 3]) == "NCWH"
        )

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same as the old one"):
            guess_new_layout("NCHW", [1, 3, 256, 256], [1, 256, 256])

    def test_element_mismatch(self):
        with pytest.raises(ValueError, match="same elements"):
            guess_new_layout("NCHW", [1, 3, 256, 256], [1, 3, 256, 128])
