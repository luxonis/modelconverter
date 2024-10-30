import os
from typing import Dict, Final

import requests


class Request:
    URL: Final[str] = "http://models.stg.hubai/models"
    # URL: Final[str] = "http://hub-stg.luxonis.com"
    HUBAI_API_KEY: Final[str] = os.getenv("HUBAI_API_KEY")
    assert HUBAI_API_KEY, "HUBAI_API_KEY is not set"
    HEADERS: Final[Dict[str, str]] = {
        "Content-Type": "application/json",
        "accept": "application/json",
        # "Authorization": f"Bearer {HUBAI_API_KEY}",
    }

    @staticmethod
    def get(endpoint: str = "", **kwargs) -> requests.Response:
        return requests.get(
            f"{Request.URL}/{endpoint.lstrip('/')}",
            headers=Request.HEADERS,
            **kwargs,
        )

    @staticmethod
    def post(endpoint: str = "", **kwargs) -> requests.Response:
        return requests.post(
            f"{Request.URL}/{endpoint}", headers=Request.HEADERS, **kwargs
        )

    @staticmethod
    def delete(endpoint: str = "", **kwargs) -> requests.Response:
        return requests.delete(
            f"{Request.URL}/{endpoint}", headers=Request.HEADERS, **kwargs
        )
