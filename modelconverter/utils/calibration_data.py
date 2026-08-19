"""Collection of the images used to calibrate a quantized model.

Quantizing a model for RVC3, RVC4 or Hailo needs a set of
representative images. This module turns whatever the config points at
-- a local directory, a remote file, archive or directory, or a view of
a Luxonis Data Format dataset -- into a local directory of images the
exporters can read.
"""

import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from luxonis_ml.data import LuxonisDataset, LuxonisLoader

from .constants import CALIBRATION_DIR, LOADERS
from .exceptions import ModelconverterException, exit_with
from .filesystem_utils import (
    download_from_remote,
    get_protocol,
    resolve_input_path,
)
from .image import read_calib_dir


def read_img_dir(path: Path, max_images: int) -> list[Path]:
    """Return the calibration images found in a directory.

    Exits the process if the directory contains no images.

    Args:
        path: Directory to read the images from.
        max_images: Maximum number of images to use. A negative value
            means all of them.

    Returns:
        Paths of the images to calibrate with.

    """
    imgs = read_calib_dir(path)
    if not imgs:
        exit_with(FileNotFoundError(f"No images found in {path}"))
    if max_images >= 0:
        logger.info(
            f"Using [{max_images}/{len(imgs)}] images for calibration."
        )
        imgs = imgs[:max_images]
    else:
        logger.info(f"Using [{len(imgs)}] images for calibration.")
    return imgs


def _find_content_root(path: Path) -> Path:
    ignored_entries = {"__MACOSX", "Thumbs.db", "desktop.ini"}

    current = path
    while True:
        contents = [
            p
            for p in current.iterdir()
            if p.name not in ignored_entries and not p.name.startswith(".")
        ]
        subdirs = [p for p in contents if p.is_dir()]
        files = [p for p in contents if p.is_file()]

        if files:
            return current
        if len(subdirs) == 1:
            current = subdirs[0]
        else:
            # Multiple subdirectories and no files. Return current and let read_img_dir raise an error
            return current


def _get_from_remote(string: str, dest: Path, max_images: int = -1) -> Path:
    path = download_from_remote(string, dest, max_images)
    if path.suffix == ".zip":
        extracted_path = path.with_suffix("")
        if extracted_path.exists():
            shutil.rmtree(extracted_path)
        extracted_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)
        return _find_content_root(extracted_path)
    return path


def download_calibration_data(string: str, max_images: int = -1) -> Path:
    """Resolve a calibration data specification to a local directory.

    Args:
        string: What to calibrate with. Either a remote URL, which is
            downloaded and, if it is a ``.zip``, extracted; a path to a
            local directory; or an LDF dataset specification of the
            form ``<dataset_name>:<split>`` or
            ``<dataset_name>:<split>:<loader_plugin>``.
        max_images: Maximum number of files to download from a remote
            directory. A negative value means no limit.

    Returns:
        Path to the local directory holding the calibration images.

    Raises:
        ModelconverterException: If the local path exists but is not a
            directory, or if the string is neither an existing path nor
            a well-formed LDF specification.

    """
    protocol = get_protocol(string)
    if protocol != "file":
        return _get_from_remote(string, CALIBRATION_DIR, max_images)

    path = Path(string).expanduser()
    if not path.exists():
        path = resolve_input_path(string) or path
    if path.exists():
        if path.is_dir():
            return path
        raise ModelconverterException(f"Path {path} is not a directory")

    match string.split(":"):
        case [dataset_name, view]:
            return load_from_ldf(dataset_name, view)
        case [dataset_name, view, loader_plugin]:
            return load_from_ldf(dataset_name, view, loader_plugin)
        case _:
            raise ModelconverterException(
                "LDF specification should be in the form <dataset_name>:<split> or <dataset_name>:<split>:<loader_plugin>"
            )


def load_from_ldf(
    dataset_name: str, view: str, loader_plugin: str | None = None
) -> Path:
    """Write the images of an LDF dataset view out as calibration data.

    The images are saved as PNGs into a per-dataset subdirectory of the
    calibration data directory.

    Args:
        dataset_name: Name of the ``LuxonisDataset`` to load. With
            ``loader_plugin`` given, it only names the directory the
            images are written to.
        view: Name of the dataset view (split) to load.
        loader_plugin: Name of a loader registered in the ``loaders``
            registry to use instead of ``LuxonisLoader``.

    Returns:
        Path to the directory the images were written to.

    Raises:
        NotImplementedError: If the dataset has more than one input.

    """
    calibration_data_dir = CALIBRATION_DIR / f"{dataset_name}"
    calibration_data_dir.mkdir(parents=True, exist_ok=True)
    if loader_plugin:
        loader = LOADERS.get(loader_plugin)(view=view)
    else:
        dataset = LuxonisDataset(dataset_name)
        loader = LuxonisLoader(dataset, view=view)

    for i, (img_arr, _) in enumerate(loader):  # type: ignore
        if not isinstance(img_arr, np.ndarray):
            raise NotImplementedError(
                "Multi input LDF datasets are not yet "
                "supported for calibration data"
            )
        img_path = calibration_data_dir / f"{i}.png"
        cv2.imwrite(str(img_path), img_arr)

    return calibration_data_dir
