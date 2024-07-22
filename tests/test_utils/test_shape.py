from modelconverter.utils.shape import Shape


def test_shape():
    shape = Shape([1, 3, 256, 256], ["N", "C", "H", "W"])
    assert shape[0] == 1
    assert shape["N"] == 1
    assert len(shape) == 4
    assert shape.guess_new_layout([3, 256, 256, 1]).layout == [
        "C",
        "H",
        "W",
        "N",
    ]
    assert shape.guess_new_layout([1, 256, 256, 3]).layout == [
        "N",
        "H",
        "W",
        "C",
    ]
    assert shape.guess_new_layout([1, 3, 256, 256]) == shape


def test_shape_complex():
    shape = Shape(
        [1, 2, 3, 4, 5, 5, 5, 6], ["N", "C1", "C2", "C3", "W", "H", "D", "C4"]
    )
    assert len(shape) == 8
    assert shape[0] == 1
    assert shape["N"] == 1
    assert shape["C1"] == 2
    assert shape["C4"] == 6
    assert shape[-1] == shape["C4"] == 6
    assert shape.guess_new_layout([1, 2, 3, 4, 5, 5, 5, 6]) == shape
    assert shape.guess_new_layout([6, 5, 5, 1, 2, 3, 5, 4]).layout == [
        "C4",
        "W",
        "H",
        "N",
        "C1",
        "C2",
        "D",
        "C3",
    ]
