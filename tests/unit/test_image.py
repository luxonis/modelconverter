"""Host-side unit tests for ``modelconverter.utils.image``.

Re-homes the golden-image coverage from the old
``tests/test_utils/test_image.py`` and extends it to hit every branch of
``read_image`` / ``read_calib_dir``.
"""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from PIL import Image

from modelconverter.utils.exceptions import ModelconverterException
from modelconverter.utils.image import read_calib_dir, read_image
from modelconverter.utils.types import DataType, Encoding, ResizeMethod
from tests.helpers.strategies import image_sizes

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


def test_layout_drives_image_dimensions(work_dir: Path):
    src = _make_img(work_dir / "src.png", size=(64, 48))

    img = read_image(
        src,
        [1, 3, 48, 64],
        Encoding.RGB,
        ResizeMethod.RESIZE,
        transpose=False,
        layout="NCHW",
    )

    assert img.shape == (48, 64, 3)


@pytest.mark.parametrize("shape", [[1], [1, 2, 3, 4, 5]])
def test_invalid_shape_raises(work_dir: Path, shape: list[int]):
    src = _make_img(work_dir / "src.png")
    with pytest.raises(ModelconverterException, match="is invalid"):
        read_image(src, shape, Encoding.RGB, ResizeMethod.RESIZE)


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


# --- Properties -----------------------------------------------------
#
# Calibration data is fed to the vendor quantizers as a raw buffer, so a
# single off-by-one in any of the three resize paths corrupts every
# sample without raising anything. The properties below fix the output
# geometry for arbitrary source and target sizes.

NUMERIC_DATA_TYPES = [
    DataType.UINT8,
    DataType.INT8,
    DataType.INT32,
    DataType.FLOAT16,
    DataType.FLOAT32,
]


@given(
    source=image_sizes,
    target=image_sizes,
    channels=st.sampled_from([1, 3]),
    resize=st.sampled_from(list(ResizeMethod)),
    transpose=st.booleans(),
)
def test_output_geometry_matches_the_requested_shape(
    work_dir: Path,
    source: tuple[int, int],
    target: tuple[int, int],
    channels: int,
    resize: ResizeMethod,
    transpose: bool,
):
    """Whatever the source size, the sample has the shape the model
    asked for."""
    src = _make_img(work_dir / "src.png", size=source)
    height, width = target
    encoding = Encoding.GRAY if channels == 1 else Encoding.RGB

    img = read_image(
        src,
        [1, channels, height, width],
        encoding,
        resize,
        transpose=transpose,
    )

    assert img.shape == (
        (channels, height, width) if transpose else (height, width, channels)
    )


@given(
    source=image_sizes,
    target=image_sizes,
    data_type=st.sampled_from(NUMERIC_DATA_TYPES),
    resize=st.sampled_from(list(ResizeMethod)),
)
def test_requested_data_type_is_always_honoured(
    work_dir: Path,
    source: tuple[int, int],
    target: tuple[int, int],
    data_type: DataType,
    resize: ResizeMethod,
):
    src = _make_img(work_dir / "src.png", size=source)
    height, width = target
    img = read_image(
        src, [height, width, 3], Encoding.RGB, resize, data_type=data_type
    )
    assert img.dtype == data_type.as_numpy_dtype()


@given(source=image_sizes, target=image_sizes)
def test_pad_centres_the_image_and_keeps_its_aspect_ratio(
    work_dir: Path, source: tuple[int, int], target: tuple[int, int]
):
    """``PAD`` letterboxes rather than stretching.

    A solid white source stays solid white through any resampling
    filter, so the non-black region *is* the scaled image: its size
    gives the aspect ratio and its offset gives the centring.
    """
    src_w, src_h = source
    height, width = target
    src = _make_img(work_dir / "src.png", size=source, color=(255, 255, 255))

    if src_w / src_h > width / height:
        expected_w, expected_h = width, round(src_h * width / src_w)
    else:
        expected_h, expected_w = height, round(src_w * height / src_h)
    assume(expected_w >= 1 and expected_h >= 1)

    img = read_image(
        src,
        [height, width, 3],
        Encoding.RGB,
        ResizeMethod.PAD,
        transpose=False,
    )

    rows = np.flatnonzero(img.any(axis=(1, 2)))
    cols = np.flatnonzero(img.any(axis=(0, 2)))
    assert (rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1) == (
        expected_h,
        expected_w,
    )
    assert (rows[0], cols[0]) == (
        (height - expected_h) // 2,
        (width - expected_w) // 2,
    )


@given(source=image_sizes, target=image_sizes)
def test_crop_takes_the_centre_of_the_source(
    work_dir: Path, source: tuple[int, int], target: tuple[int, int]
):
    """``CROP`` copies pixels through untouched, no resampling."""
    src_w, src_h = source
    height, width = target
    # A crop wider or taller than the source would be black-padded by
    # PIL; that case is covered by the shape property above.
    assume(width <= src_w and height <= src_h)

    # A non-constant pattern, so a wrong offset cannot go unnoticed.
    y, x = np.indices((src_h, src_w))
    pattern = np.stack([(y * 7 + x * 13) % 256] * 3, axis=-1).astype(np.uint8)
    src = work_dir / "src.png"
    Image.fromarray(pattern).save(src)

    img = read_image(
        src,
        [height, width, 3],
        Encoding.RGB,
        ResizeMethod.CROP,
        transpose=False,
    )

    left = int(src_w / 2 - width / 2)
    upper = int(src_h / 2 - height / 2)
    expected = pattern[upper : upper + height, left : left + width]
    assert np.array_equal(img, expected)


@given(
    source=image_sizes,
    target=image_sizes,
    resize=st.sampled_from(list(ResizeMethod)),
)
def test_bgr_is_rgb_with_the_channels_reversed(
    work_dir: Path,
    source: tuple[int, int],
    target: tuple[int, int],
    resize: ResizeMethod,
):
    src_w, src_h = source
    y, x = np.indices((src_h, src_w))
    pattern = np.stack(
        [(y * 7) % 256, (x * 13) % 256, (y + x) % 256], axis=-1
    ).astype(np.uint8)
    src = work_dir / "src.png"
    Image.fromarray(pattern).save(src)

    height, width = target
    shape = [height, width, 3]
    rgb = read_image(src, shape, Encoding.RGB, resize, transpose=False)
    bgr = read_image(src, shape, Encoding.BGR, resize, transpose=False)

    assert np.array_equal(bgr, rgb[..., ::-1])
