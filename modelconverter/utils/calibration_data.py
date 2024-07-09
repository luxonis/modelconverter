import logging
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

import cv2
from luxonis_ml.data import LuxonisDataset, LuxonisLoader

from .constants import CALIBRATION_DIR, LOADERS, SHARED_DIR
from .exceptions import ModelconverterException, exit_with
from .filesystem_utils import download_from_remote, get_protocol
from .image import read_calib_dir

logger = logging.getLogger(__name__)


def read_img_dir(path: Path, max_images: int) -> List[Path]:
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


def _get_from_remote(string: str, dest: Path, max_images: int = -1) -> Path:
    path = download_from_remote(string, dest, max_images)
    if path.suffix == ".zip":
        extracted_path = path.with_suffix("")
        if extracted_path.exists():
            shutil.rmtree(extracted_path)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path.parent)
        return extracted_path
    return path


def download_calibration_data(string: str, max_images: int = -1) -> Path:
    protocol = get_protocol(string)
    if protocol != "file":
        return _get_from_remote(string, CALIBRATION_DIR, max_images)

    path = Path(string)
    if not path.exists():
        path = SHARED_DIR / string
    if path.exists():
        if path.is_dir():
            return path
        else:
            raise ModelconverterException(f"Path {path} is not a directory")
    try:
        try_dataset_split = string.split(":")
        if len(try_dataset_split) == 2:
            dataset_name, view = try_dataset_split
            loader_plugin = None
        elif len(try_dataset_split) == 3:
            dataset_name, view, loader_plugin = try_dataset_split
        else:
            raise ModelconverterException(
                "LDF specification should be in the form <dataset_name>:<split> or <dataset_name>:<split>:<loader_plugin>"
            )
    except ValueError as e:
        raise ModelconverterException(
            f"{string} is either in an unsupported format "
            "or points to a non-existing directory"
        ) from e
    return load_from_ldf(dataset_name, view, loader_plugin)


def load_from_ldf(
    dataset_name: str, view: str, loader_plugin: Optional[str] = None
) -> Path:
    calibration_data_dir = CALIBRATION_DIR / f"{dataset_name}"
    calibration_data_dir.mkdir(parents=True, exist_ok=True)
    if loader_plugin:
        loader = LOADERS.get(loader_plugin)(view=view)
    else:
        dataset = LuxonisDataset(dataset_name)
        loader = LuxonisLoader(dataset, view=view)

    for i, (img_arr, _) in enumerate(loader):  # type: ignore
        img_path = calibration_data_dir / f"{i}.png"
        cv2.imwrite(str(img_path), img_arr)

    return calibration_data_dir
