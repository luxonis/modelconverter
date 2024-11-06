from typing import Dict, Final

import requests
from requests import HTTPError, Response

from modelconverter.utils import environ


class Request:
    URL: Final[str] = f"{environ.HUBAI_URL.rstrip('/')}/api/v1"
    HEADERS: Final[Dict[str, str]] = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {environ.HUBAI_API_KEY}",
    }

    @staticmethod
    def _check_response(response: Response) -> Response:
        if response.status_code != 200:
            raise HTTPError(response.json())
        return response

    @staticmethod
    def get(endpoint: str = "", **kwargs) -> requests.Response:
        return Request._check_response(
            requests.get(
                Request._get_url(endpoint),
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
                Request._get_url(endpoint), headers=headers, **kwargs
            )
        )

    @staticmethod
    def delete(endpoint: str = "", **kwargs) -> requests.Response:
        return Request._check_response(
            requests.delete(
                Request._get_url(endpoint), headers=Request.HEADERS, **kwargs
            )
        )

    @staticmethod
    def _get_url(endpoint: str) -> str:
        return f"{Request.URL}/{endpoint.lstrip('/')}".rstrip("/")
