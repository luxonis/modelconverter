from collections.abc import Mapping
from json import JSONDecodeError

import requests
from luxonis_ml.typing import ParamValue
from requests import HTTPError, Response

from modelconverter.utils import environ


class Request:
    @staticmethod
    def url() -> str:
        return f"{environ.HUBAI_URL.rstrip('/')}/models/api/v1"

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
    def _process_response(response: Response) -> ParamValue:
        return Request._get_json(Request._check_response(response))

    @staticmethod
    def _check_response(response: Response) -> Response:
        if response.status_code >= 400:
            raise HTTPError(Request._get_json(response), response=response)
        return response

    @staticmethod
    def _get_json(response: Response) -> ParamValue:
        try:
            return response.json()
        except JSONDecodeError as e:
            raise HTTPError(
                f"Unexpected response from the server:\n{response.text}",
                response=response,
            ) from e

    @staticmethod
    def get(endpoint: str = "", **kwargs) -> ParamValue:
        return Request._process_response(
            requests.get(
                Request._get_url(endpoint),
                headers=Request.headers(),
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def get_records(
        endpoint: str = "", **kwargs
    ) -> list[dict[str, ParamValue]]:
        """The response as the list of records a Hub endpoint returns.

        Every listing endpoint answers with an array of objects, so the
        shape is checked once here instead of at each call site.
        """
        data = Request.get(endpoint, **kwargs)
        if not isinstance(data, list):
            raise HTTPError(f"Expected a list of records from `{endpoint}`.")
        records = [item for item in data if isinstance(item, Mapping)]
        if len(records) != len(data):
            raise HTTPError(f"Expected only objects from `{endpoint}`.")
        return [dict(record) for record in records]

    @staticmethod
    def dag_get(endpoint: str = "", **kwargs) -> ParamValue:
        return Request._process_response(
            requests.get(
                Request._get_url(endpoint, Request.dag_url()),
                headers=Request.headers(),
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def post(endpoint: str = "", **kwargs) -> ParamValue:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**Request.headers(), **kwargs.pop("headers")}
        return Request._process_response(
            requests.post(
                Request._get_url(endpoint),
                headers=headers,
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def delete(endpoint: str = "", **kwargs) -> ParamValue:
        return Request._process_response(
            requests.delete(
                Request._get_url(endpoint),
                headers=Request.headers(),
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def put(endpoint: str = "", **kwargs) -> ParamValue:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**headers, **kwargs.pop("headers")}
        return Request._process_response(
            requests.put(
                Request._get_url(endpoint),
                headers=headers,
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def patch(endpoint: str = "", **kwargs) -> ParamValue:
        headers = Request.headers()
        if "headers" in kwargs:
            headers = {**headers, **kwargs.pop("headers")}
        return Request._process_response(
            requests.patch(
                Request._get_url(endpoint),
                headers=headers,
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def _get_url(endpoint: str, base_url: str | None = None) -> str:
        base_url = base_url or Request.url()
        return f"{base_url}/{endpoint.lstrip('/')}".rstrip("/")
