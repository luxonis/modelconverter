import multiprocessing
from contextlib import suppress

import keyring
from luxonis_ml.utils import Environ as BaseEnviron
from pydantic import model_validator
from typing_extensions import Self


def get_password_with_timeout(
    service_name: str, username: str, timeout: float = 5
) -> str | None:  # pragma: no cover
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
    # The cache is hidden and managed for the user, so it also has to stop
    # growing on its own. Staged inputs are whole models and calibration
    # sets, so the budget has to be generous enough that a normal working
    # set survives; `0` turns the eviction off.
    MODELCONVERTER_CACHE_MAX_SIZE: str = "50GiB"

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
