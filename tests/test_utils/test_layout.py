from modelconverter.utils.layout import guess_new_layout, make_default_layout


def test_shape():
    old_shape = [1, 3, 256, 256]
    old_layout = "NCHW"
    assert guess_new_layout(old_layout, old_shape, [3, 256, 256, 1]) == "CHWN"
    assert guess_new_layout(old_layout, old_shape, [1, 256, 256, 3]) == "NHWC"
    assert guess_new_layout(old_layout, old_shape, [1, 3, 256, 256]) == "NCHW"


def test_shape_complex():
    old_shape = [1, 2, 3, 4, 5, 5, 5, 6]
    old_layout = "NABCWHDE"
    assert (
        guess_new_layout(old_layout, old_shape, [1, 2, 3, 4, 5, 5, 5, 6])
        == "NABCWHDE"
    )
    assert (
        guess_new_layout(old_layout, old_shape, [6, 5, 5, 1, 2, 3, 5, 4])
        == "EWHNABDC"
    )


def test_make_default_layout():
    assert make_default_layout([1, 3, 256, 256]) == "NCHW"
    assert make_default_layout([1, 19, 7, 8]) == "NCDE"
    assert make_default_layout([256, 256, 3]) == "HWC"
