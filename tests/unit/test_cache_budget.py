"""Tests for the size budget the cache is kept within.

The cache is hidden from the user and filled without being asked, so it
also has to stop growing on its own. Eviction is least-recently-used and
never touches an entry a conversion may still be reading, which is what
these tests are about.
"""

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from modelconverter.utils import input_staging
from modelconverter.utils.constants import CONTAINER_SHARED_DIR
from modelconverter.utils.environ import environ

# Sets the cache budget for the duration of one test.
SetBudget = Callable[[str], None]


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Rebases the staging cache into the test's temp directory."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache)
    return cache


@pytest.fixture
def budget(monkeypatch: pytest.MonkeyPatch) -> SetBudget:
    def set_budget(value: str) -> None:
        monkeypatch.setattr(
            environ, "MODELCONVERTER_CACHE_MAX_SIZE", value, raising=False
        )

    return set_budget


def _stage(src: Path, cache_dir: Path, released: bool = True) -> Path:
    """Stages ``src`` and, by default, drops the claim on it.

    A released entry stands for one staged by a conversion that has since
    finished, which is the only kind the sweep may evict.
    """
    entry = input_staging._stage_file(src, cache_dir / "inputs").parent
    if released:
        for marker in entry.glob(f"{input_staging._IN_USE_PREFIX}*"):
            marker.unlink()
    return entry


def _age(entry: Path, used: float) -> None:
    (entry / input_staging._USED_MARKER).write_text(str(used))


def _model(tmp_path: Path, name: str, size: int) -> Path:
    """Create a file of exactly ``size`` bytes, distinct from every other one.

    Entries are keyed by content, so two models of the same size have to
    differ in their bytes to be two entries at all.
    """
    src = tmp_path / name
    src.write_bytes((name.encode() * size)[:size])
    return src


def test_the_least_recently_used_entry_is_evicted_first(
    tmp_path: Path, cache_dir: Path, budget: SetBudget
) -> None:
    old = _stage(_model(tmp_path, "old.tflite", 4096), cache_dir)
    recent = _stage(_model(tmp_path, "recent.tflite", 4096), cache_dir)
    _age(old, 1000.0)
    _age(recent, 2000.0)
    budget("4KiB")

    input_staging.enforce_cache_budget()

    assert not old.exists()
    assert recent.exists()


def test_eviction_stops_once_the_cache_fits(
    tmp_path: Path, cache_dir: Path, budget: SetBudget
) -> None:
    """The budget is a ceiling, not a target: everything that still fits
    under it is kept, however old it is.
    """
    entries = []
    for index in range(4):
        entry = _stage(_model(tmp_path, f"{index}.tflite", 1024), cache_dir)
        _age(entry, 1000.0 + index)
        entries.append(entry)
    budget("3KiB")

    input_staging.enforce_cache_budget()

    assert [entry.exists() for entry in entries] == [False, True, True, True]


def test_an_entry_in_use_survives_over_budget(
    tmp_path: Path, cache_dir: Path, budget: SetBudget
) -> None:
    """A conversion in another terminal is still reading its inputs."""
    claimed = _stage(
        _model(tmp_path, "claimed.tflite", 4096), cache_dir, released=False
    )
    _age(claimed, 1000.0)
    budget("1KiB")

    input_staging.enforce_cache_budget()

    assert claimed.exists()


def test_entries_staged_by_this_run_survive(
    tmp_path: Path, cache_dir: Path, budget: SetBudget
) -> None:
    """The conversion about to start needs exactly what it just staged,
    even when the claim is somehow missing.
    """
    entry = _stage(_model(tmp_path, "model.tflite", 4096), cache_dir)
    budget("1KiB")

    input_staging.enforce_cache_budget()

    assert entry.exists()


def test_an_unlimited_budget_evicts_nothing(
    tmp_path: Path, cache_dir: Path, budget: SetBudget
) -> None:
    entry = _stage(_model(tmp_path, "model.tflite", 4096), cache_dir)
    _age(entry, 1000.0)
    budget("0")

    input_staging.enforce_cache_budget()

    assert entry.exists()


def test_an_unreadable_budget_is_ignored(
    tmp_path: Path, cache_dir: Path, budget: SetBudget
) -> None:
    """A typo in the environment must not start deleting models, nor
    stop the conversion.
    """
    entry = _stage(_model(tmp_path, "model.tflite", 4096), cache_dir)
    _age(entry, 1000.0)
    budget("as much as it takes")

    input_staging.enforce_cache_budget()

    assert entry.exists()


def test_downloads_are_evicted_alongside_staged_inputs(
    cache_dir: Path, budget: SetBudget
) -> None:
    """The container downloads remote models straight into the cache,
    and those are the entries nothing host-side ever staged.
    """
    download = cache_dir / "models" / "hub-model.onnx"
    download.parent.mkdir(parents=True)
    download.write_bytes(b"x" * 4096)
    os.utime(download, (1000.0, 1000.0))
    budget("1KiB")

    input_staging.enforce_cache_budget()

    assert not download.exists()


def test_downloads_survive_while_another_conversion_runs(
    cache_dir: Path, budget: SetBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download has no claim of its own -- the container writes it
    under the launcher's claim on the cache root -- so nothing may be
    taken from underneath a run in another terminal.
    """
    download = cache_dir / "models" / "hub-model.onnx"
    download.parent.mkdir(parents=True)
    download.write_bytes(b"x" * 4096)
    os.utime(download, (1000.0, 1000.0))
    # A claim on the cache root belonging to some other live process.
    other = subprocess.Popen([sys.executable, "-c", "input()"], stdin=-1)
    monkeypatch.setattr(input_staging, "_is_in_use", _claimed_by(other.pid))
    budget("1KiB")

    try:
        input_staging.enforce_cache_budget()
    finally:
        other.kill()
        other.wait()

    assert download.exists()


def _claimed_by(pid: int) -> Callable[..., bool]:
    def is_in_use(entry: Path, ignore_pid: int | None = None) -> bool:
        return entry.name == "cache" and ignore_pid != pid

    return is_in_use


def test_the_digest_memo_is_neither_counted_nor_evicted(
    cache_dir: Path, budget: SetBudget
) -> None:
    """Rebuilding it means re-reading every staged model, which costs
    more than the handful of bytes it holds.
    """
    memo = cache_dir / "digests"
    memo.mkdir(parents=True)
    (memo / "abcdef").write_text("0123456789abcdef")
    budget("1B")

    input_staging.enforce_cache_budget()

    assert (memo / "abcdef").exists()


def test_an_entry_claimed_mid_sweep_is_put_back(
    tmp_path: Path, cache_dir: Path
) -> None:
    """The window between deciding to evict and deleting is real: a
    staging that reused the entry claims it in between, and the claim is
    on the entry wherever it has been renamed to.
    """
    src = _model(tmp_path, "model.tflite", 4096)
    entry = _stage(src, cache_dir)

    # The sweep found no claim; by the time the entry is renamed aside,
    # another run has staged the same input and claimed it.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(input_staging, "_is_in_use", lambda *_: True)
        assert input_staging._evict(entry) is False

    assert (entry / "model.tflite").read_bytes() == src.read_bytes()


def test_the_leftovers_of_a_killed_sweep_are_cleaned_up(
    cache_dir: Path, budget: SetBudget
) -> None:
    """A sweep killed between renaming an entry aside and deleting it
    would otherwise leave a full-sized entry nothing ever looks at.
    """
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    inputs = cache_dir / "inputs"
    trash = inputs / f"{input_staging._TRASH_PREFIX}{dead.pid}-abcdef"
    trash.mkdir(parents=True)
    (trash / "model.tflite").write_bytes(b"x" * 4096)
    budget("1KiB")

    input_staging.enforce_cache_budget()

    assert not trash.exists()


def test_a_running_sweep_keeps_its_own_trash(
    cache_dir: Path, budget: SetBudget
) -> None:
    """Another sweep is mid-eviction; its leftovers are not ours to
    delete.
    """
    inputs = cache_dir / "inputs"
    trash = inputs / f"{input_staging._TRASH_PREFIX}{os.getpid()}-abcdef"
    trash.mkdir(parents=True)
    (trash / "model.tflite").write_bytes(b"x" * 4096)
    budget("1KiB")

    input_staging.enforce_cache_budget()

    assert trash.exists()


def test_staging_trims_the_cache_it_just_added_to(
    tmp_path: Path, cache_dir: Path, budget: SetBudget
) -> None:
    """The sweep runs as part of staging, which is the only moment the
    launcher knows what the conversion needs and the container has not
    started reading it yet.
    """
    stale = _stage(_model(tmp_path, "stale.tflite", 4096), cache_dir)
    _age(stale, 1000.0)
    needed = _model(tmp_path, "needed.tflite", 4096)
    budget("4KiB")

    staged = input_staging.stage_inputs(
        ["--model-path", str(needed)], {"--model-path"}
    )

    assert not stale.exists()
    assert _host_path(staged[1], cache_dir).read_bytes() == needed.read_bytes()


def _host_path(staged: str, cache_dir: Path) -> Path:
    return cache_dir / Path(staged).relative_to(CONTAINER_SHARED_DIR)
