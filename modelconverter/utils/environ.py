import multiprocessing
from contextlib import suppress

import keyring
from luxonis_ml.utils import Environ as BaseEnviron
from pydantic import model_validator
from typing_extensions import Self


def get_password_with_timeout(
    service_name: str, username: str, timeout: float = 5
) -> str | None:
    def _get_password(q: multiprocessing.Queue) -> None:
        try:
            result = keyring.get_password(service_name, username)
            q.put(result)
        except Exception:
            q.put(None)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_get_password, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    if not q.empty():
        return q.get()
    return None


class Environ(BaseEnviron):
    HUBAI_API_KEY: str | None = None
    HUBAI_URL: str = "https://easyml.cloud.luxonis.com/"

    @model_validator(mode="after")
    def validate_hubai_api_key(self) -> Self:
        if self.HUBAI_API_KEY:
            return self

        with suppress(Exception):
            self.HUBAI_API_KEY = get_password_with_timeout(
                "ModelConverter", "api_key"
            )

        return self


environ = Environ()
