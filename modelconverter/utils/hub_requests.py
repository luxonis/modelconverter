"""Thin HTTP client for the Luxonis HubAI API.

Wraps ``requests`` with the HubAI base URL and API key taken from the
environment, parses the JSON body of every response and turns failures
into ``requests.HTTPError``. It backs the parts of ModelConverter that
accept a hub model instead of a local file, such as resolving a model
variant to the archive to convert or to benchmark.
"""

from json import JSONDecodeError
from typing import Any

import requests
from requests import HTTPError, Response

from modelconverter.utils import environ


class Request:
    """Namespace of helpers for calling the HubAI API."""

    @staticmethod
    def url() -> str:
        """Return the base URL of the HubAI models API."""
        return f"{environ.HUBAI_URL.rstrip('/')}/models/api/v1"

    @staticmethod
    def dag_url() -> str:
        """Return the base URL of the HubAI DAGs API."""
        return f"{environ.HUBAI_URL.rstrip('/')}/dags/api/v1"

    @staticmethod
    def headers() -> dict[str, str]:
        """Return the headers every HubAI request is sent with.

        Returns:
            The ``accept`` header and the bearer ``Authorization``
            header built from the configured API key.

        Raises:
            ValueError: If no HubAI API key is configured.

        """
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
        """Send a ``GET`` request to the HubAI models API.

        Args:
            endpoint: Endpoint appended to the base URL.
            **kwargs: Additional keyword arguments forwarded to
                ``requests.get``.

        Returns:
            The parsed JSON body of the response.

        Raises:
            HTTPError: If the server responds with an error status or
                with a body that is not valid JSON.

        """
        return Request._process_response(
            requests.get(
                Request._get_url(endpoint),
                headers=Request.headers(),
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def dag_get(endpoint: str = "", **kwargs) -> Any:
        """Send a ``GET`` request to the HubAI DAGs API.

        Args:
            endpoint: Endpoint appended to the DAGs base URL.
            **kwargs: Additional keyword arguments forwarded to
                ``requests.get``.

        Returns:
            The parsed JSON body of the response.

        Raises:
            HTTPError: If the server responds with an error status or
                with a body that is not valid JSON.

        """
        return Request._process_response(
            requests.get(
                Request._get_url(endpoint, Request.dag_url()),
                headers=Request.headers(),
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def post(endpoint: str = "", **kwargs) -> Any:
        """Send a ``POST`` request to the HubAI models API.

        Args:
            endpoint: Endpoint appended to the base URL.
            **kwargs: Additional keyword arguments forwarded to
                ``requests.post``. A ``headers`` entry is merged into
                the default headers instead of replacing them.

        Returns:
            The parsed JSON body of the response.

        Raises:
            HTTPError: If the server responds with an error status or
                with a body that is not valid JSON.

        """
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
    def delete(endpoint: str = "", **kwargs) -> Any:
        """Send a ``DELETE`` request to the HubAI models API.

        Args:
            endpoint: Endpoint appended to the base URL.
            **kwargs: Additional keyword arguments forwarded to
                ``requests.delete``.

        Returns:
            The parsed JSON body of the response.

        Raises:
            HTTPError: If the server responds with an error status or
                with a body that is not valid JSON.

        """
        return Request._process_response(
            requests.delete(
                Request._get_url(endpoint),
                headers=Request.headers(),
                timeout=200,
                **kwargs,
            )
        )

    @staticmethod
    def put(endpoint: str = "", **kwargs) -> Any:
        """Send a ``PUT`` request to the HubAI models API.

        Args:
            endpoint: Endpoint appended to the base URL.
            **kwargs: Additional keyword arguments forwarded to
                ``requests.put``. A ``headers`` entry is merged into
                the default headers instead of replacing them.

        Returns:
            The parsed JSON body of the response.

        Raises:
            HTTPError: If the server responds with an error status or
                with a body that is not valid JSON.

        """
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
    def patch(endpoint: str = "", **kwargs) -> Any:
        """Send a ``PATCH`` request to the HubAI models API.

        Args:
            endpoint: Endpoint appended to the base URL.
            **kwargs: Additional keyword arguments forwarded to
                ``requests.patch``. A ``headers`` entry is merged into
                the default headers instead of replacing them.

        Returns:
            The parsed JSON body of the response.

        Raises:
            HTTPError: If the server responds with an error status or
                with a body that is not valid JSON.

        """
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
