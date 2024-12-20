from contextlib import suppress
from typing import Optional

import keyring
from luxonis_ml.utils import Environ as BaseEnviron
from pydantic import model_validator
from typing_extensions import Self


class Environ(BaseEnviron):
    HUBAI_API_KEY: Optional[str] = None
    HUBAI_URL: str = "https://easyml.cloud.luxonis.com/models/"

    @model_validator(mode="after")
    def validate_hubai_api_key(self) -> Self:
        if self.HUBAI_API_KEY:
            return self

        with suppress(Exception):
            self.HUBAI_API_KEY = keyring.get_password(
                "ModelConverter", "api_key"
            )

        return self


environ = Environ()
