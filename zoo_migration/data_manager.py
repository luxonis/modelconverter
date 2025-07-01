import csv
import os
import tarfile
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from b2sdk.v2 import B2Api
from loguru import logger
from requests import HTTPError, Response
from tqdm import tqdm

from modelconverter.hub.__main__ import (
    instance_create,
    upload,
)
from modelconverter.utils import environ
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    MISC_DIR,
)


class Request:
    @staticmethod
    def url() -> str:
        return f"{environ.HUBAI_URL.rstrip('/')}/datasets/api/v1"

    @staticmethod
    def dag_url() -> str:
        return f"{environ.HUBAI_URL.rstrip('/')}/dags/api/v1"

    @staticmethod
    def headers() -> dict[str, str]:
        if environ.HUBAI_API_KEY is None:
            raise ValueError("HUBAI_API_KEY is not set")

        return {
            "accept": "application/json",
            "Authorization": f"Bearer {environ.HUBAI_API_KEY}",
        }

    @staticmethod
    def _process_response(response: Response) -> Any:
        return Request._get_json(Request._check_response(response))

    @staticmethod
    def _check_response(response: Response) -> Response:
        if response.status_code >= 400:
            raise HTTPError(Request._get_json(response), response=response)
        return response

    @staticmethod
    def _get_json(response: Response) -> Any:
        try:
            return response.json()
        except JSONDecodeError as e:
            raise HTTPError(
                f"Unexpected response from the server:\n{response.text}",
                response=response,
            ) from e

    @staticmethod
    def get(endpoint: str = "", **kwargs) -> Any:
        return Request._process_response(
            requests.get(
                Request._get_url(endpoint),
                headers=Request.headers(),
                timeout=10,
                **kwargs,
            )
        )

    @staticmethod
    def dag_get(endpoint: str = "", **kwargs) -> Any:
        return Request._process_response(
            requests.get(
                Request._get_url(endpoint, Request.dag_url()),
                headers=Request.headers(),
                timeout=10,
                **kwargs,
            )
        )

    @staticmethod
    def post(endpoint: str = "", **kwargs) -> Any:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**Request.headers(), **kwargs.pop("headers")}
        return Request._process_response(
            requests.post(
                Request._get_url(endpoint),
                headers=headers,
                timeout=10,
                **kwargs,
            )
        )

    @staticmethod
    def delete(endpoint: str = "", **kwargs) -> Any:
        return Request._process_response(
            requests.delete(
                Request._get_url(endpoint),
                headers=Request.headers(),
                timeout=10,
                **kwargs,
            )
        )

    @staticmethod
    def put(endpoint: str = "", **kwargs) -> Any:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**headers, **kwargs.pop("headers")}
        return Request._process_response(
            requests.put(
                Request._get_url(endpoint),
                headers=headers,
                timeout=10,
                **kwargs,
            )
        )

    @staticmethod
    def patch(endpoint: str = "", **kwargs) -> Any:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**headers, **kwargs.pop("headers")}
        return Request._process_response(
            requests.patch(
                Request._get_url(endpoint),
                headers=headers,
                timeout=10,
                **kwargs,
            )
        )

    @staticmethod
    def _get_url(endpoint: str, base_url: str | None = None) -> str:
        base_url = base_url or Request.url()
        return f"{base_url}/{endpoint.lstrip('/')}".rstrip("/")


def request(query: str) -> list[str]:
    # This should return a list of image URLs based on the input `query`
    # Here's a sample dummy list:
    out = Request.get(f"datasets/{query}?with_samples=true")
    return [i["media_link"] for i in out["samples"]]


def safe_extract(
    tar_path: Path, dest: Path, *, numeric_owner: bool = False
) -> None:
    """Extract *tar_path* into *dest* but refuse path-traversal
    members."""
    dest = dest.resolve()
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = dest / member.name
            if not member_path.resolve().is_relative_to(dest):
                raise ValueError(
                    f"Blocked path-traversal file: {member.name!r}"
                )
        tar.extractall(dest, numeric_owner=numeric_owner)  # noqa: S202


def download_dataset(
    dataset_id: str, download_path: Path = CALIBRATION_DIR / "datasets"
) -> Path:
    save_dir = download_path / dataset_id
    if Path.exists(save_dir):
        logger.info(
            f"Dataset {dataset_id} already exists in {save_dir}. Skipping download."
        )
        return save_dir

    logger.info(f"Downloading dataset {dataset_id} from Luxonis Hub")
    try:
        image_links = request(dataset_id)
    except Exception as e:
        logger.exception(f"Failed to request dataset {dataset_id}: {e}")

    Path.mkdir(save_dir, exist_ok=True)
    for i, url in tqdm(enumerate(image_links)):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            suffix = Path(urlparse(url).path).suffix or ".jpg"

            image_path = save_dir / f"image_{i + 1}{suffix}"
            with open(image_path, "wb") as img_file:
                img_file.write(response.content)
        except Exception as e:
            logger.error(f"Failed to download image {i + 1} from {url}: {e}")
            continue

    return save_dir


def download_datasets_from_csv(
    csv_file: str, base_download_path: Path = CALIBRATION_DIR / "datasets"
) -> None:
    """Downloads datasets based on a CSV file containing dataset IDs.

    The CSV should have dataset IDs in the first column.
    """

    Path.mkdir(base_download_path, exist_ok=True)

    with open(csv_file, newline="") as f:
        reader = csv.reader(f)
        _ = next(reader)  # remove header row

        for _row_num, row in enumerate(reader, start=2):
            if len(row) < 4:
                continue

            quant_folder_name = row[2].strip()
            test_folder_name = row[3].strip()

            for folder_name in [quant_folder_name, test_folder_name]:
                if folder_name in [
                    "/",
                    "General Dataset from Hub",
                    "Driving Dataset from Hub",
                    "/ todo?",
                ]:
                    continue

                download_dataset(folder_name, base_download_path)


def download_snpe_files(
    version: str, download_directory: Path = MISC_DIR
) -> None:
    """Downloads SNPE files for the specified version from backblaze
    bucket.

    Requires the B2 environment variables to be set:
    B2_ACCOUNT_ID, B2_APPLICATION_KEY, B2_BUCKET_NAME.
    The files are downloaded to modelconverter.utils.constants.MISC_DIR /snpe_version

    Parameters
    ----------
    version : str
        The version of SNPE to download. Format: snpe_x_x_x
    """
    logger.info(
        f"Downloading SNPE files for version {version} to {download_directory}"
    )

    if Path.exists(download_directory / f"{version}.tar.xz"):
        logger.info(
            f"SNPE files for version '{version}' already exist in '{download_directory / version}.tar.xz'. Skipping download."
        )
        return

    b2_application_key_id = os.getenv("B2_APPLICATION_KEY_ID", None)
    b2_application_key = os.getenv("B2_APPLICATION_KEY", None)

    if not b2_application_key_id or not b2_application_key:
        logger.error(
            "B2_ACCOUNT_ID and B2_APPLICATION_KEY_ID must be set in environment variables."
        )
        raise ValueError

    api = B2Api()
    api.authorize_account(
        "production", b2_application_key_id, b2_application_key
    )

    bucket = api.get_bucket_by_name("luxonis")

    download_directory.mkdir(parents=True, exist_ok=True)
    downloaded_file = bucket.download_file_by_name(
        "modelconverter/snpe_" + version + ".tar.xz"
    )
    downloaded_file.save_to(download_directory / f"{version}.tar.xz")
    safe_extract(download_directory / f"{version}.tar.xz", download_directory)

    logger.info(
        f"Downloaded and extracted SNPE files for version {version} to {download_directory / version}"
    )


def upload_new_instance(
    instance_params: dict[str, Any], archive: Path
) -> None:
    logger.info("Creating new instance")
    instance = instance_create(**instance_params, silent=True)
    logger.info(f"New instance created: {instance['id']}, {instance['name']}")
    upload(str(archive), instance["id"])
