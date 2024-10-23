import os
from typing import Dict, Final

import requests

# assert HUBAI_API_KEY, "HUBAI_API_KEY is not set"


class Request:
    URL: Final[str] = "https://easyml.stg.hubcloud/models"
    HUBAI_API_KEY: Final[str] = os.getenv("HUBAI_API_KEY")
    HEADERS: Final[Dict[str, str]] = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {HUBAI_API_KEY}",
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
