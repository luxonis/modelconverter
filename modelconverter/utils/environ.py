from typing import Optional

from luxonis_ml.utils import Environ as BaseEnviron


class Environ(BaseEnviron):
    HUBAI_API_KEY: Optional[str] = None
    HUBAI_URL: str = "https://easyml.cloud.luxonis.com/models/"


environ = Environ()