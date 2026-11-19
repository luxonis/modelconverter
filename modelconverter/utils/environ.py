"""Environment-backed settings of ModelConverter.

Extends the ``luxonis_ml`` environment with the settings the converter
adds on top of it: the API key and the address of the HubAI instance
models are fetched from, and the size budget of the local input cache.
The single `environ` instance defined here is what the rest of the
package reads.
"""

import json
import subprocess
import sys
from contextlib import suppress

from luxonis_ml.typing import Params
from luxonis_ml.utils import Environ as BaseEnviron
from pydantic import model_validator


def get_password_with_timeout(
    service_name: str, username: str, timeout: float = 5
) -> str | None:
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
    # Starting a multiprocessing fork after logging has created threads can
    # deadlock on Python 3.12: the fork warning is routed through Loguru while
    # its at-fork locks are held. A fresh interpreter avoids those locks and
    # does not re-import this module's eager Environ instance.
    script = (
        "import json, sys, keyring; "
        "print(json.dumps(keyring.get_password(sys.argv[1], sys.argv[2])))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, service_name, username],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        password = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return password if isinstance(password, str) else None


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

    @model_validator(mode="before")
    @classmethod
    def validate_hubai_api_key(cls, data: Params) -> Params:
        """Fall back to the system keyring for the HubAI API key.

        The keyring is only consulted when the key was not provided
        through the environment. Any failure of the lookup is ignored
        and leaves the key unset.

        Args:
            data: Raw settings collected from the environment.

        Returns:
            The settings, with ``HUBAI_API_KEY`` filled in if the
            keyring held one.

        """
        if data.get("HUBAI_API_KEY"):
            return data

        api_key = data.get("HUBAI_API_KEY")
        with suppress(Exception):
            api_key = get_password_with_timeout("ModelConverter", "api_key")

        return {**data, "HUBAI_API_KEY": api_key}


environ = Environ()
