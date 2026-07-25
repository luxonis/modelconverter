import os

import pytest
from luxonis_ml.utils import setup_logging

os.environ.setdefault("LUXONIS_TELEMETRY_ENABLED", "false")

setup_logging()


_PLATFORM_MARKERS = frozenset({"rvc2", "rvc3", "rvc4", "hailo"})


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-mark unit tests and auto-select the current platform.

    Host-side unit tests (``tests/unit``) get the ``unit`` marker.
    Conversion tests carry an explicit per-platform marker (``rvc2`` /
    ``rvc3`` / ``rvc4`` / ``hailo``) applied at parametrization, so
    ``pytest -m rvc2`` selects only the RVC2 conversions.

    When running inside a platform Docker image, ``MODELCONVERTER_TARGET``
    is set (injected by the launcher), so conversion tests for the *other*
    platforms are auto-deselected. That way ``modelconverter shell rvc4
    --dev -c "pytest -x"`` runs exactly the unit + RVC4 conversion tests
    without needing an explicit ``-m``.
    """
    for item in items:
        if f"{os.sep}unit{os.sep}" in str(item.fspath):
            item.add_marker("unit")

    target = os.getenv("MODELCONVERTER_TARGET")
    if not target:
        return

    selected, deselected = [], []
    for item in items:
        item_platforms = _PLATFORM_MARKERS.intersection(
            marker.name for marker in item.iter_markers()
        )
        # Keep unit tests (no platform marker) and this platform's tests.
        if not item_platforms or target in item_platforms:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
