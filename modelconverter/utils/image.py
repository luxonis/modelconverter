"""Loading of calibration and inference input data.

Calibration data is provided as a directory of images or of raw arrays.
This module finds those files and turns a single one of them into the
array shape, color encoding and data type that the exporters and
inferers of the individual platforms feed to their models.
"""

from itertools import chain
from pathlib import Path

import numpy as np
from luxonis_ml.typing import PathType
from PIL import Image, ImageOps

from modelconverter.utils.exceptions import ModelconverterException
from modelconverter.utils.types import DataType, Encoding, ResizeMethod


def read_image(
    path: PathType,
    shape: list[int],
    encoding: Encoding,
    resize_method: ResizeMethod,
    data_type: DataType | None = None,
    transpose: bool = True,
    layout: str | None = None,
) -> np.ndarray:
    """Read a single calibration or inference input from a file.

    ``.npy`` files are loaded as they are and ``.raw`` files are read as
    a flat buffer of ``data_type`` reshaped to ``shape``. Any other file
    is opened as an image, converted to ``encoding``, and fitted to the
    height and width taken from ``shape``.

    Args:
        path: Path of the file to read.
        shape: Shape of the model input the file is read for.
        encoding: Color encoding the image is converted to. Ignored for
            ``.npy`` and ``.raw`` files.
        resize_method: How the image is fitted to the requested height
            and width. Ignored for ``.npy`` and ``.raw`` files.
        data_type: Data type the resulting array is cast to. Images are
            cast to ``uint8`` if it is ``None``. Required for ``.raw``
            files and ignored for ``.npy`` files.
        transpose: If ``True``, the image is transposed from ``HWC`` to
            ``CHW``. Ignored for ``.npy`` and ``.raw`` files.
        layout: Lettercode layout of ``shape``, used to find the ``H``,
            ``W``, and ``C`` axes. If it is ``None``, its length does
            not match ``shape``, or it lacks ``H`` or ``W``, the axes
            are guessed from the number of dimensions.

    Returns:
        The loaded data as a ``numpy.ndarray``.

    Raises:
        ModelconverterException: If a ``.raw`` file is read without
            ``data_type``, or if ``shape`` has a length that cannot be
            interpreted as an image.

    """
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)

    if path.suffix == ".raw":
        if data_type is None:
            raise ModelconverterException(
                "Input data type must be specified when"
                "using `.raw` files for calibration."
            )
        return np.fromfile(path, dtype=data_type.as_numpy_dtype()).reshape(
            shape
        )

    h, w, c = _image_size(shape, layout)
    img = _fit_image(
        _convert_color(Image.open(path), encoding), w, h, resize_method
    )
    img_arr = np.array(img)
    if data_type is not None:
        img_arr = img_arr.astype(data_type.as_numpy_dtype())
    else:
        img_arr = img_arr.astype(np.uint8)
    if encoding == Encoding.GRAY or c == 1:
        img_arr = img_arr[..., np.newaxis]
    if transpose:
        img_arr = img_arr.transpose(2, 0, 1)
    return img_arr


def read_calib_dir(path: Path) -> list[Path]:
    """Collect the calibration files in a directory.

    Args:
        path: Directory to search. It is not searched recursively.

    Returns:
        Paths of the ``.jpg``, ``.png``, ``.jpeg``, ``.npy``, and
        ``.raw`` files in the directory.

    """
    return list(
        chain(
            *[
                path.glob(suffix)
                for suffix in [
                    "*.jpg",
                    "*.png",
                    "*.jpeg",
                    "*.npy",
                    "*.raw",
                ]
            ]
        )
    )


def _image_size(shape: list[int], layout: str | None) -> tuple[int, int, int]:
    """Read the height, width and channel count from an input shape.

    Without a usable layout, the axes are guessed from the number of
    dimensions.
    """
    if (
        layout is not None
        and len(layout) == len(shape)
        and "H" in layout
        and "W" in layout
    ):
        # The layout tells us which axes are spatial vs. channel, so a
        # channels-last (e.g. TFLite NHWC) shape is read correctly.
        c = shape[layout.index("C")] if "C" in layout else 1
        return shape[layout.index("H")], shape[layout.index("W")], c
    if len(shape) == 2:
        return shape[0], shape[1], 1
    if len(shape) == 3 and shape[0] == 1:
        return shape[1], shape[2], 1
    if len(shape) == 3:
        return shape[0], shape[1], shape[2]
    if len(shape) == 4:
        return shape[2], shape[3], shape[1]
    raise ModelconverterException(
        f"Input shape `{shape}` is invalid for an image. "
        "Use `.npy` or `.raw` files as calibration data instead."
    )


def _convert_color(img: Image.Image, encoding: Encoding) -> Image.Image:
    """Convert an image to the channel order of ``encoding``.

    Any other encoding keeps the image as it is.
    """
    if encoding == Encoding.BGR:
        return Image.fromarray(np.array(img.convert("RGB"))[..., ::-1])
    if encoding == Encoding.RGB:
        return img.convert("RGB")
    if encoding == Encoding.GRAY:
        return img.convert("L")
    return img


def _fit_image(
    img: Image.Image, w: int, h: int, resize_method: ResizeMethod
) -> Image.Image:
    """Fit an image to a width and height with ``resize_method``.

    ``PAD`` is the remaining method.
    """
    if resize_method == ResizeMethod.CROP:
        return ImageOps.fit(img, (w, h), method=Image.Resampling.BICUBIC)
    if resize_method == ResizeMethod.CENTER_CROP_NO_RESIZE:
        left = int(img.size[0] / 2 - w / 2)
        upper = int(img.size[1] / 2 - h / 2)
        return img.crop((left, upper, left + w, upper + h))
    if resize_method == ResizeMethod.RESIZE:
        return img.resize((w, h))
    return _pad_image(img, w, h)


def _pad_image(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resize an image to fit inside ``w`` by ``h`` and pad it with black.

    The image keeps its aspect ratio and sits in the center.
    """
    orig_ratio = img.size[0] / img.size[1]
    new_ratio = w / h
    if orig_ratio > new_ratio:
        # If original image is wider, resize by width
        scale_factor = w / img.size[0]
        resized_img = img.resize((w, round(img.size[1] * scale_factor)))
    else:
        # If original image is taller, resize by height
        scale_factor = h / img.size[1]
        resized_img = img.resize((round(img.size[0] * scale_factor), h))

    new_img = Image.new(img.mode, (w, h), "black")
    ulc = ((w - resized_img.size[0]) // 2, (h - resized_img.size[1]) // 2)
    new_img.paste(resized_img, ulc)
    return new_img
