from pathlib import Path

import numpy as np
from PIL import Image

from modelconverter.utils.image import read_calib_dir, read_image
from modelconverter.utils.types import Encoding, ResizeMethod

DATA_PATH = Path("tests/data/test_utils/test_image")


def read_test_image(path: Path, mode: str):
    return np.array(Image.open(path).convert(mode), dtype=np.uint8)


orig_img = read_test_image(DATA_PATH / "orig.jpg", "RGB")
crop_img = read_test_image(DATA_PATH / "crop.png", "RGB")
crop_gray_img = read_test_image(DATA_PATH / "crop_gray.png", "L").reshape(
    256, 256, 1
)
pad_img = read_test_image(DATA_PATH / "pad.png", "RGB")
resized_img = read_test_image(DATA_PATH / "resized.png", "RGB")
resized_bgr_img = read_test_image(DATA_PATH / "resized_bgr.png", "RGB")


def test_read_calib_dir():
    read_files = sorted([path.name for path in read_calib_dir(DATA_PATH)])
    expected_files = sorted([path.name for path in DATA_PATH.iterdir()])
    assert read_files == expected_files


def assert_image_equal(img1, img2):
    assert img1.shape == img2.shape
    assert np.allclose(img1, img2)


def read_and_compare(expected: np.ndarray, *args, **kwargs):
    img = read_image(*args, **kwargs)
    assert_image_equal(img, expected)


def test_read_resize():
    read_and_compare(
        resized_img,
        DATA_PATH / "orig.jpg",
        [256, 256, 3],
        Encoding.RGB,
        ResizeMethod.RESIZE,
        transpose=False,
    )


def test_read_transposed():
    read_and_compare(
        resized_img.transpose(2, 0, 1),
        DATA_PATH / "orig.jpg",
        [256, 256, 3],
        Encoding.RGB,
        ResizeMethod.RESIZE,
        transpose=True,
    )


def test_read_resize_bgr():
    read_and_compare(
        resized_bgr_img,
        DATA_PATH / "orig.jpg",
        [256, 256, 3],
        Encoding.BGR,
        ResizeMethod.RESIZE,
        transpose=False,
    )


def test_read_pad():
    read_and_compare(
        pad_img,
        DATA_PATH / "orig.jpg",
        [256, 256, 3],
        Encoding.RGB,
        ResizeMethod.PAD,
        transpose=False,
    )


def test_read_crop():
    read_and_compare(
        crop_img,
        DATA_PATH / "orig.jpg",
        [256, 256, 3],
        Encoding.RGB,
        ResizeMethod.CROP,
        transpose=False,
    )


def test_read_crop_gray():
    read_and_compare(
        crop_gray_img,
        DATA_PATH / "orig.jpg",
        [256, 256, 1],
        Encoding.GRAY,
        ResizeMethod.CROP,
        transpose=False,
    )


def test_read_different():
    img = read_image(
        DATA_PATH / "orig.jpg",
        [256, 256, 3],
        Encoding.RGB,
        ResizeMethod.RESIZE,
        transpose=False,
    )
    assert not np.allclose(img, crop_img)
