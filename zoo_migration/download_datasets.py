from json import JSONDecodeError
from typing import Any

import requests
from requests import HTTPError, Response

from modelconverter.utils import environ


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
                **kwargs,
            )
        )

    @staticmethod
    def dag_get(endpoint: str = "", **kwargs) -> Any:
        return Request._process_response(
            requests.get(
                Request._get_url(endpoint, Request.dag_url()),
                headers=Request.headers(),
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
                Request._get_url(endpoint), headers=headers, **kwargs
            )
        )

    @staticmethod
    def delete(endpoint: str = "", **kwargs) -> Any:
        return Request._process_response(
            requests.delete(
                Request._get_url(endpoint), headers=Request.headers(), **kwargs
            )
        )

    @staticmethod
    def put(endpoint: str = "", **kwargs) -> Any:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**headers, **kwargs.pop("headers")}
        return Request._process_response(
            requests.put(Request._get_url(endpoint), headers=headers, **kwargs)
        )

    @staticmethod
    def patch(endpoint: str = "", **kwargs) -> Any:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**headers, **kwargs.pop("headers")}
        return Request._process_response(
            requests.patch(
                Request._get_url(endpoint), headers=headers, **kwargs
            )
        )

    @staticmethod
    def _get_url(endpoint: str, base_url: str | None = None) -> str:
        base_url = base_url or Request.url()
        return f"{base_url}/{endpoint.lstrip('/')}".rstrip("/")


import csv
import os

from tqdm import tqdm


# Dummy implementation of request(), replace with your actual logic
def request(query):
    # This should return a list of image URLs based on the input `query`
    # Here's a sample dummy list:
    out = Request.get(f"datasets/{query}?with_samples=true")
    return [i["media_link"] for i in out["samples"]]


# === CONFIGURATION ===
csv_file = "mappings.csv"  # Replace with your actual file
base_download_path = (
    "shared_with_container/calibration_data/datasets"  # Or any path you want
)

os.makedirs(base_download_path, exist_ok=True)

with open(csv_file, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header

    for row_num, row in enumerate(reader, start=2):
        # Start from row 2 (first data row)
        if len(row) < 4:
            print(f"Row {row_num} is too short: {row}")
            continue

        model_name = row[0].strip()
        quant_folder_name = row[2].strip()
        test_folder_name = row[3].strip()

        print(model_name)

        folder_names = [quant_folder_name, test_folder_name]

        for folder_name in folder_names:
            if folder_name in [
                "/",
                "General Dataset from Hub",
                "Driving Dataset from Hub",
                "/ todo?",
            ]:
                print(f"Skipped {folder_name} for {model_name}")
                continue

            try:
                image_links = request(folder_name)
            except Exception as e:
                print(f"ERROR during request: {e}")
                continue

            save_dir = os.path.join(base_download_path, folder_name)
            os.makedirs(save_dir, exist_ok=True)

            for i, url in tqdm(enumerate(image_links)):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    ext = os.path.splitext(url)[1].split("?")[0] or ".jpg"
                    image_path = os.path.join(save_dir, f"image_{i + 1}{ext}")
                    with open(image_path, "wb") as img_file:
                        img_file.write(response.content)
                except Exception as e:
                    print(
                        f"Failed to download image from {url} (row {row_num}): {e}"
                    )
