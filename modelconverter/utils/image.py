from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from modelconverter.utils.exceptions import ModelconverterException
from modelconverter.utils.types import DataType, Encoding, ResizeMethod


def read_image(
    path: Union[str, Path],
    shape: List[int],
    encoding: Encoding,
    resize_method: ResizeMethod,
    data_type: Optional[DataType] = None,
    transpose: bool = True,
) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
        return arr

    if path.suffix == ".raw":
        if data_type is None:
            raise ModelconverterException(
                "Input data type must be specified when"
                "using `.raw` files for calibration."
            )
        arr = np.fromfile(path, dtype=data_type.as_numpy_dtype()).reshape(
            shape
        )
        return arr

    if len(shape) == 2:
        h, w, c = *shape, 1
    elif len(shape) == 3:
        if shape[0] == 1:
            _, h, w, c = *shape, 1
        else:
            h, w, c = shape
    elif len(shape) == 4:
        _, c, h, w = shape
    else:
        raise ModelconverterException(
            f"Input shape `{shape}` is invalid for an image. "
            "Use `.npy` or `.raw` files as calibration data instead."
        )
    img = Image.open(path)
    if encoding == Encoding.BGR:
        img = img.convert("RGB")
        img = Image.fromarray(np.array(img)[..., ::-1])
    elif encoding == Encoding.RGB:
        img = img.convert("RGB")
    elif encoding == Encoding.GRAY:
        img = img.convert("L")
    if resize_method == ResizeMethod.CROP:
        left = int(img.size[0] / 2 - w / 2)
        upper = int(img.size[1] / 2 - h / 2)
        right = left + w
        lower = upper + h
        img = img.crop((left, upper, right, lower))
    elif resize_method == ResizeMethod.RESIZE:
        img = img.resize((w, h))
    elif resize_method == ResizeMethod.PAD:
        orig_ratio = img.size[0] / img.size[1]

        # Calculate aspect ratio of new size
        new_ratio = w / h

        # Compare aspects
        if orig_ratio > new_ratio:
            # If original image is wider, resize by width
            scale_factor = w / img.size[0]
            new_height = round(img.size[1] * scale_factor)
            resized_img = img.resize((w, new_height))
        else:
            # If original image is taller, resize by height
            scale_factor = h / img.size[1]
            new_width = round(img.size[0] * scale_factor)
            resized_img = img.resize((new_width, h))

        # Create a new, blank image with padding color
        new_img = Image.new(img.mode, (w, h), "black")

        # Paste resized image into center of new, blank image
        ulc = ((w - resized_img.size[0]) // 2, (h - resized_img.size[1]) // 2)
        new_img.paste(resized_img, ulc)
        img = new_img
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


def read_calib_dir(path: Path) -> List[Path]:
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
