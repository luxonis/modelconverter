from itertools import product
from typing import Final

import pytest

from modelconverter.utils import subprocess_run

URL_PREFIX: Final[str] = "shared_with_container/configs/"


@pytest.mark.parametrize(
    "from_format, to_format, model",
    [
        (f, t, m)
        for (f, t, m) in product(
            ["nn_archive", "native"],
            ["nn_archive", "native"],
            ["resnet18", "yolov8n_seg"],
        )
        if (f, t, m) != ("native", "nn_archive", "yolov8n_seg")
    ],
)
def test_convert(from_format: str, to_format: str, model: str):
    if from_format == "nn_archive":
        url = f"{URL_PREFIX}{model}.tar.xz"
    else:
        url = f"{URL_PREFIX}{model}.yaml"

    result = subprocess_run(
        f"modelconverter convert rvc4 --dev --path {url} --to {to_format}"
    )
    assert result.returncode == 0, result.stderr + result.stdout
