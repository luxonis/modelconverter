"""Host-side unit tests for ``modelconverter.utils.image``.

Re-homes the golden-image coverage from the old
``tests/test_utils/test_image.py`` and extends it to hit every branch of
``read_image`` / ``read_calib_dir``.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modelconverter.utils.exceptions import ModelconverterException
from modelconverter.utils.image import read_calib_dir, read_image
from modelconverter.utils.types import DataType, Encoding, ResizeMethod

# Absolute so the autouse cwd-isolation fixture cannot break the path.
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "test_utils"
    / "test_image"
)
ORIG = DATA_DIR / "orig.jpg"


def _golden(name: str, mode: str = "RGB") -> np.ndarray:
    return np.array(Image.open(DATA_DIR / name).convert(mode), dtype=np.uint8)


def _make_img(
    path: Path,
    size: tuple[int, int] = (64, 48),
    color: tuple[int, int, int] = (10, 20, 30),
) -> Path:
    """Write a solid-color RGB image (``size`` is ``(width,
    height)``)."""
    Image.new("RGB", size, color).save(path)
    return path


# --------------------------------------------------------------------------- #
# read_image: golden comparisons (reused from the old suite)                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("golden", "encoding", "resize"),
    [
        ("resized.png", Encoding.RGB, ResizeMethod.RESIZE),
        ("resized_bgr.png", Encoding.BGR, ResizeMethod.RESIZE),
        ("crop.png", Encoding.RGB, ResizeMethod.CROP),
        ("pad.png", Encoding.RGB, ResizeMethod.PAD),
    ],
)
def test_golden_rgb(golden: str, encoding: Encoding, resize: ResizeMethod):
    expected = _golden(golden)
    img = read_image(ORIG, [256, 256, 3], encoding, resize, transpose=False)
    assert img.shape == expected.shape
    assert img.dtype == np.uint8
    assert np.allclose(img, expected)


def test_golden_gray_crop():
    expected = _golden("crop_gray.png", "L").reshape(256, 256, 1)
    img = read_image(
        ORIG, [256, 256, 1], Encoding.GRAY, ResizeMethod.CROP, transpose=False
    )
    assert img.shape == expected.shape
    assert np.allclose(img, expected)


def test_transpose_true():
    expected = _golden("resized.png").transpose(2, 0, 1)
    img = read_image(
        ORIG, [256, 256, 3], Encoding.RGB, ResizeMethod.RESIZE, transpose=True
    )
    assert img.shape == expected.shape
    assert np.allclose(img, expected)


def test_resize_differs_from_crop():
    img = read_image(
        ORIG, [256, 256, 3], Encoding.RGB, ResizeMethod.RESIZE, transpose=False
    )
    assert not np.allclose(img, _golden("crop.png"))


# --------------------------------------------------------------------------- #
# read_image: PAD aspect-ratio branches                                       #
# --------------------------------------------------------------------------- #


def test_pad_wide_image():
    # orig.jpg is 640x428 -> wider than square target (orig_ratio > new_ratio).
    img = read_image(
        ORIG, [256, 256, 3], Encoding.RGB, ResizeMethod.PAD, transpose=False
    )
    assert img.shape == (256, 256, 3)
    # Wide image -> top/bottom rows are the black padding.
    assert np.all(img[0] == 0)
    assert np.all(img[-1] == 0)


def test_pad_tall_image(work_dir: Path):
    tall = _make_img(work_dir / "tall.png", size=(100, 300))
    img = read_image(
        tall, [256, 256, 3], Encoding.RGB, ResizeMethod.PAD, transpose=False
    )
    assert img.shape == (256, 256, 3)
    # Tall image -> left/right columns are the black padding.
    assert np.all(img[:, 0] == 0)
    assert np.all(img[:, -1] == 0)


# --------------------------------------------------------------------------- #
# read_image: shape-length unpacking branches                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("shape", "encoding", "expected_shape"),
    [
        # len 2 -> h, w, c=1 (grayscale keeps it 3D via newaxis)
        ([128, 96], Encoding.GRAY, (128, 96, 1)),
        # len 3 with leading 1 -> _, h, w, c=1
        ([1, 128, 96], Encoding.GRAY, (128, 96, 1)),
        # len 3 otherwise -> h, w, c
        ([128, 96, 3], Encoding.RGB, (128, 96, 3)),
        # len 4 -> _, c, h, w
        ([1, 3, 128, 96], Encoding.RGB, (128, 96, 3)),
    ],
)
def test_shape_branches(
    work_dir: Path,
    shape: list[int],
    encoding: Encoding,
    expected_shape: tuple[int, int, int],
):
    src = _make_img(work_dir / "src.png", size=(64, 48))
    img = read_image(
        src, shape, encoding, ResizeMethod.RESIZE, transpose=False
    )
    assert img.shape == expected_shape


@pytest.mark.parametrize("shape", [[1], [1, 2, 3, 4, 5]])
def test_invalid_shape_raises(work_dir: Path, shape: list[int]):
    src = _make_img(work_dir / "src.png")
    with pytest.raises(ModelconverterException, match="is invalid"):
        read_image(src, shape, Encoding.RGB, ResizeMethod.RESIZE)


# --------------------------------------------------------------------------- #
# read_image: dtype handling                                                  #
# --------------------------------------------------------------------------- #


def test_dtype_cast():
    img = read_image(
        ORIG,
        [256, 256, 3],
        Encoding.RGB,
        ResizeMethod.RESIZE,
        data_type=DataType.FLOAT32,
        transpose=False,
    )
    assert img.dtype == np.float32


def test_dtype_default_uint8():
    img = read_image(
        ORIG, [256, 256, 3], Encoding.RGB, ResizeMethod.RESIZE, transpose=False
    )
    assert img.dtype == np.uint8


# --------------------------------------------------------------------------- #
# read_image: .npy and .raw paths                                             #
# --------------------------------------------------------------------------- #


def test_read_npy(work_dir: Path):
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    npy = work_dir / "data.npy"
    np.save(npy, arr)
    out = read_image(npy, [2, 3, 4], Encoding.RGB, ResizeMethod.RESIZE)
    assert out.dtype == arr.dtype
    assert np.array_equal(out, arr)


def test_read_raw(work_dir: Path):
    arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    raw = work_dir / "data.raw"
    arr.tofile(raw)
    out = read_image(
        raw,
        [2, 3, 4],
        Encoding.RGB,
        ResizeMethod.RESIZE,
        data_type=DataType.UINT8,
    )
    assert out.shape == (2, 3, 4)
    assert np.array_equal(out, arr)


def test_read_raw_without_data_type_raises(work_dir: Path):
    raw = work_dir / "data.raw"
    np.arange(6, dtype=np.uint8).tofile(raw)
    with pytest.raises(ModelconverterException, match="data type"):
        read_image(raw, [2, 3], Encoding.RGB, ResizeMethod.RESIZE)


# --------------------------------------------------------------------------- #
# read_calib_dir                                                              #
# --------------------------------------------------------------------------- #


def test_read_calib_dir_matches_contents():
    read_files = sorted(p.name for p in read_calib_dir(DATA_DIR))
    expected = sorted(p.name for p in DATA_DIR.iterdir())
    assert read_files == expected


def test_read_calib_dir_filters_and_globs(work_dir: Path):
    calib = work_dir / "calib"
    calib.mkdir()
    _make_img(calib / "a.jpg")
    _make_img(calib / "b.png")
    np.save(calib / "c.npy", np.zeros(1))
    np.zeros(1, dtype=np.uint8).tofile(calib / "d.raw")
    (calib / "notes.txt").write_text("ignored")

    found = sorted(p.name for p in read_calib_dir(calib))
    assert found == ["a.jpg", "b.png", "c.npy", "d.raw"]


def test_read_calib_dir_empty(work_dir: Path):
    empty = work_dir / "empty"
    empty.mkdir()
    assert read_calib_dir(empty) == []
