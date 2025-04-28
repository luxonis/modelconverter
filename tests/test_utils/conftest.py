import shutil

import pytest

from .test_modifier import DATA_DIR


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
