import shutil

import pytest

from .test_modifier import DATA_DIR


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
