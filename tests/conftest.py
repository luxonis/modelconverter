import os

import pytest
from luxonis_ml.utils import setup_logging

os.environ.setdefault("LUXONIS_TELEMETRY_ENABLED", "false")

setup_logging()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-mark tests by directory: ``tests/unit`` -> ``unit``,
    ``tests/e2e`` -> ``e2e``."""
    for item in items:
        path = str(item.fspath)
        if f"{os.sep}unit{os.sep}" in path:
            item.add_marker("unit")
        elif f"{os.sep}e2e{os.sep}" in path:
            item.add_marker("e2e")
