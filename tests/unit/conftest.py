"""Fixtures for Tier-1 host-side unit tests.

Every test runs inside its own ``tmp_path`` working directory. Because
the ``modelconverter.utils.constants`` ``*_DIR`` values are *relative*
paths (``shared_with_container/...``), ``monkeypatch.chdir`` re-roots
all of them into the temp dir, so nothing ever writes into the repo's
real ``shared_with_container/`` and tests are safe under ``pytest -n
auto``.
"""

from pathlib import Path

import pytest

from tests.helpers.onnx_factory import standard_dummy_onnx

_SHARED_SUBDIRS = ["misc", "configs", "outputs", "calibration_data", "models"]


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chdir into an isolated temp dir and pre-create the shared
    dirs."""
    monkeypatch.chdir(tmp_path)
    shared = tmp_path / "shared_with_container"
    for sub in _SHARED_SUBDIRS:
        (shared / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """The isolated working directory (same as cwd for the test)."""
    return tmp_path


@pytest.fixture
def dummy_onnx(tmp_path: Path) -> Path:
    """Absolute path to the canonical two-input dummy ONNX model.

    Absolute so ``resolve_path`` short-circuits on ``Path.exists()`` and
    no remote download is attempted.
    """
    return standard_dummy_onnx(
        tmp_path / "shared_with_container" / "models" / "dummy_model.onnx"
    ).resolve()
