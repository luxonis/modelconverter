from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Optional, Union

import keyring
from luxonis_ml.utils import Environ as BaseEnviron
from pydantic import AliasChoices, Field, model_validator
from typing_extensions import Annotated, Self


def get_password_with_timeout(
    service_name: str, username: str, timeout: Union[float, int] = 5
):
    def _get_password():
        return keyring.get_password(service_name, username)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_get_password)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return None


class Environ(BaseEnviron):
    HUBAI_API_KEY: Annotated[
        Optional[str],
        Field(
            validation_alias=AliasChoices("HUBAI_API_KEY", "HUB_AI_API_KEY")
        ),
    ] = None
    HUBAI_URL: Annotated[
        str, Field(validation_alias=AliasChoices("HUBAI_URL", "HUB_AI_URL"))
    ] = "https://easyml.cloud.luxonis.com/models/"

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
