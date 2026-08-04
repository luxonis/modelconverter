"""Fixtures for Tier-1 host-side unit tests.

Every test runs inside its own ``tmp_path`` working directory with the
cache re-rooted there. The ``modelconverter.utils.constants`` ``*_DIR``
values are absolute (the hidden cache directory, ``./output``) and are
read once at import time, so ``monkeypatch.chdir`` alone would not move
them: ``_isolate_cwd`` rebinds them wherever they were imported. Nothing
ever writes into the real ``~/.cache/modelconverter`` and tests are safe
under ``pytest -n auto``.
"""

import sys
from pathlib import Path

import pytest

from modelconverter.utils import constants
from tests.helpers.onnx_factory import standard_dummy_onnx


def _redirect_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Points every ``*_DIR`` constant at ``tmp_path``.

    The constants are plain module-level values, so a module doing ``from
    ... import MODELS_DIR`` holds its own reference; patching only
    ``constants`` would leave those -- including the ones the test
    modules themselves imported -- pointing at the real cache.
    """
    # Exactly the layout `get_cache_dir` produces for this `XDG_CACHE_HOME`,
    # so a test may patch either one and still land in the same place.
    cache_home = tmp_path / ".cache"
    cache = cache_home / "modelconverter"
    replacements = {
        "SHARED_DIR": cache,
        "MISC_DIR": cache / "misc",
        "CONFIGS_DIR": cache / "configs",
        "CALIBRATION_DIR": cache / "calibration_data",
        "MODELS_DIR": cache / "models",
        "INPUTS_DIR": cache / "inputs",
        "OUTPUTS_DIR": tmp_path / "output",
    }
    for path in replacements.values():
        path.mkdir(parents=True, exist_ok=True)

    originals = {name: getattr(constants, name) for name in replacements}
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith(("modelconverter", "tests")):
            continue
        for name, replacement in replacements.items():
            if getattr(module, name, None) == originals[name]:
                monkeypatch.setattr(module, name, replacement)

    # Anything calling `get_cache_dir()` at run time rather than reading the
    # constants -- the container launcher, input staging -- follows along.
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    monkeypatch.setattr(constants, "get_cache_dir", lambda: cache)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chdir into an isolated temp dir and re-root the cache into it."""
    monkeypatch.chdir(tmp_path)
    _redirect_dirs(monkeypatch, tmp_path)
    return tmp_path


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """The isolated working directory (same as cwd for the test)."""
    return tmp_path


@pytest.fixture
def dummy_onnx() -> Path:
    """Absolute path to the canonical two-input dummy ONNX model.

    Absolute so ``resolve_path`` short-circuits on ``Path.exists()`` and
    no remote download is attempted.
    """
    return standard_dummy_onnx(
        constants.MODELS_DIR / "dummy_model.onnx"
    ).resolve()
