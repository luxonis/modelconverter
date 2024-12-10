from json import JSONDecodeError
from typing import Any, Dict, Final, Optional

import requests
from requests import HTTPError, Response

from modelconverter.utils import environ


class Request:
    URL: Final[str] = f"{environ.HUBAI_URL.rstrip('/')}/models/api/v1"
    DAG_URL: Final[str] = URL.replace("models", "dags")
    HEADERS: Final[Dict[str, str]] = {
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
                headers=Request.HEADERS,
                **kwargs,
            )
        )

    @staticmethod
    def dag_get(endpoint: str = "", **kwargs) -> Any:
        return Request._process_response(
            requests.get(
                Request._get_url(endpoint, Request.DAG_URL),
                headers=Request.HEADERS,
                **kwargs,
            )
        )

    @staticmethod
    def post(endpoint: str = "", **kwargs) -> Any:
        headers = Request.HEADERS
        if "headers" in kwargs:
            headers = {**Request.HEADERS, **kwargs.pop("headers")}
        return Request._process_response(
            requests.post(
                Request._get_url(endpoint), headers=headers, **kwargs
            )
        )

    @staticmethod
    def delete(endpoint: str = "", **kwargs) -> Any:
        return Request._process_response(
            requests.delete(
                Request._get_url(endpoint), headers=Request.HEADERS, **kwargs
            )
        )

    @staticmethod
    def _get_url(endpoint: str, base_url: Optional[str] = None) -> str:
        base_url = base_url or Request.URL
        return f"{base_url}/{endpoint.lstrip('/')}".rstrip("/")
