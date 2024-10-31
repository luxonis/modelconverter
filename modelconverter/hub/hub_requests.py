from typing import Dict, Final

import requests
from requests import Response

from modelconverter.utils import environ


class Request:
    URL: Final[str] = environ.HUBAI_URL
    HEADERS: Final[Dict[str, str]] = {
        "Content-Type": "application/json",
        "accept": "application/json",
        # "Authorization": f"Bearer {environ.HUBAI_API_KEY}",
    }

    @staticmethod
    def _check_response(response: Response) -> Response:
        response.raise_for_status()
        if response.status_code != 200:
            raise Exception(response.json())
        return response

    @staticmethod
    def get(endpoint: str = "", **kwargs) -> requests.Response:
        return Request._check_response(
            requests.get(
                f"{Request.URL}/{endpoint.lstrip('/')}",
                headers=Request.HEADERS,
                **kwargs,
            )
        )

    @staticmethod
    def post(endpoint: str = "", **kwargs) -> requests.Response:
        headers = Request.HEADERS
        if "headers" in kwargs:
            headers = {**Request.HEADERS, **kwargs.pop("headers")}
        return Request._check_response(
            requests.post(
                f"{Request.URL}/{endpoint}", headers=headers, **kwargs
            )
        )

    @staticmethod
    def delete(endpoint: str = "", **kwargs) -> requests.Response:
        return Request._check_response(
            requests.delete(
                f"{Request.URL}/{endpoint}", headers=Request.HEADERS, **kwargs
            )
        )
