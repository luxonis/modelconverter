"""Environment-backed settings of ModelConverter.

Extends the ``luxonis_ml`` environment with the settings the converter
adds on top of it: the API key and the address of the HubAI instance
models are fetched from, and the size budget of the local input cache.
The single `environ` instance defined here is what the rest of the
package reads.
"""

import multiprocessing
from contextlib import suppress

import keyring
from luxonis_ml.utils import Environ as BaseEnviron
from pydantic import model_validator
from typing_extensions import Self


def get_password_with_timeout(
    service_name: str, username: str, timeout: float = 5
) -> str | None:  # pragma: no cover
    """Read a password from the system keyring, giving up on a timeout.

    The lookup runs in a separate process so that a keyring backend
    that blocks -- waiting for a locked keyring to be unlocked, for
    instance -- cannot hang the caller indefinitely.

    Args:
        service_name: Name of the keyring service the password is
            stored under.
        username: Name of the keyring user the password is stored
            under.
        timeout: How long to wait for the lookup, in seconds.

    Returns:
        The stored password, or ``None`` if the lookup timed out,
        raised, or found nothing.

    """

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
    """Settings ModelConverter reads from the environment.

    Values are taken from the process environment or a ``.env`` file,
    as with any ``luxonis_ml`` environment.

    Attributes:
        HUBAI_API_KEY: API key used to authenticate against HubAI. When
            it is not set, it is looked up in the system keyring.
        HUBAI_URL: Base URL of the HubAI instance to talk to.
        MODELCONVERTER_CACHE_MAX_SIZE: Size the local input cache is
            trimmed down to, written as a human-readable size such as
            ``50GiB``. ``0`` turns the eviction off.

    """

    HUBAI_API_KEY: str | None = None
    HUBAI_URL: str = "https://easyml.cloud.luxonis.com/"
    # The cache is hidden and managed for the user, so it also has to stop
    # growing on its own. Staged inputs are whole models and calibration
    # sets, so the budget has to be generous enough that a normal working
    # set survives; `0` turns the eviction off.
    MODELCONVERTER_CACHE_MAX_SIZE: str = "50GiB"

    @model_validator(mode="after")
    def validate_hubai_api_key(self) -> Self:
        """Fall back to the system keyring for the HubAI API key.

        The keyring is only consulted when the key was not provided
        through the environment. Any failure of the lookup is ignored
        and leaves the key unset.

        Returns:
            The environment itself, with ``HUBAI_API_KEY`` filled in if
            the keyring held one.

        """
        if self.HUBAI_API_KEY:
            return self

        with suppress(Exception):
            self.HUBAI_API_KEY = get_password_with_timeout(
                "ModelConverter", "api_key"
            )

        return self


environ = Environ()
