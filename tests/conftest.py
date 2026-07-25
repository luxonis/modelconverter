import os

import pytest
from luxonis_ml.utils import setup_logging

os.environ.setdefault("LUXONIS_TELEMETRY_ENABLED", "false")

setup_logging()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-mark host-side unit tests (``tests/unit``) with ``unit``.

    Conversion tests carry an explicit per-platform marker (``rvc2`` /
    ``rvc3`` / ``rvc4`` / ``hailo``) applied at parametrization, so
    ``pytest -m rvc2`` selects only the RVC2 conversions.
    """
    for item in items:
        if f"{os.sep}unit{os.sep}" in str(item.fspath):
            item.add_marker("unit")
