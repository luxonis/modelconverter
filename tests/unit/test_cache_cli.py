import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from modelconverter import __main__ as cli
from modelconverter.utils import input_staging


def _nonempty_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "modelconverter"
    cache.mkdir()
    (cache / "entry").write_bytes(b"cached")
    return cache


def test_cache_clean_declined_keeps_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _nonempty_cache(tmp_path)
    monkeypatch.setattr(cli, "get_cache_dir", lambda: cache)
    monkeypatch.setattr(cli.Confirm, "ask", lambda *_args, **_kwargs: False)

    cli.cache_clean()

    assert cache.exists()
    assert (cache / "entry").exists()


def test_cache_clean_confirmed_removes_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _nonempty_cache(tmp_path)
    monkeypatch.setattr(cli, "get_cache_dir", lambda: cache)
    monkeypatch.setattr(cli.Confirm, "ask", lambda *_args, **_kwargs: True)

    cli.cache_clean()

    assert not cache.exists()


def _deny_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes the cache root unlistable, as a container that was killed
    before it could hand the mounts back leaves it."""

    def denied(_self: Path) -> NoReturn:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "iterdir", denied)


def test_cache_info_reports_an_unreadable_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache = _nonempty_cache(tmp_path)
    monkeypatch.setattr(cli, "get_cache_dir", lambda: cache)
    _deny_listing(monkeypatch)

    cli.cache_info()

    assert "cannot be read" in " ".join(capsys.readouterr().out.split())


def test_cache_clean_cleans_an_unreadable_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _nonempty_cache(tmp_path)
    monkeypatch.setattr(cli, "get_cache_dir", lambda: cache)
    monkeypatch.setattr(cli.Confirm, "ask", lambda *_args, **_kwargs: True)
    _deny_listing(monkeypatch)

    cli.cache_clean()

    assert not cache.exists()


def test_cache_clean_yes_skips_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _nonempty_cache(tmp_path)
    monkeypatch.setattr(cli, "get_cache_dir", lambda: cache)

    def unexpected_prompt(*_args, **_kwargs) -> bool:
        raise AssertionError("confirmation prompt should be skipped")

    monkeypatch.setattr(cli.Confirm, "ask", unexpected_prompt)

    cli.cache_clean(yes=True)

    assert not cache.exists()


def _claimed_by(cache: Path, pid: int) -> Path:
    """A staged input entry claimed by ``pid``, as a launcher marks the
    ones its container has open."""
    entry = cache / "inputs" / "digest"
    entry.mkdir(parents=True)
    (entry / "image.jpg").write_bytes(b"calibration image")
    (entry / f"{input_staging._IN_USE_PREFIX}{pid}").touch()
    return entry


def test_cache_clean_spares_inputs_a_running_conversion_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is bind-mounted into every running container, so
    emptying it pulls the staged inputs out from under a conversion that
    is still reading them."""
    cache = _nonempty_cache(tmp_path)
    entry = _claimed_by(cache, os.getpid())
    monkeypatch.setattr(cli, "get_cache_dir", lambda: cache)
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache)

    cli.cache_clean(yes=True)

    assert (entry / "image.jpg").read_bytes() == b"calibration image"


def test_cache_clean_ignores_a_claim_from_a_dead_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that was killed leaves its marker behind; refusing forever
    on account of it would make the cache impossible to clear."""
    cache = _nonempty_cache(tmp_path)
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    _claimed_by(cache, dead.pid)
    monkeypatch.setattr(cli, "get_cache_dir", lambda: cache)
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache)

    cli.cache_clean(yes=True)

    assert not cache.exists()


def test_cache_clean_declines_when_stdin_cannot_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Piped or closed stdin makes the prompt raise ``EOFError``. Nothing
    in this command group runs inside ``catch_exceptions``, and a
    destructive command must not read silence as consent either."""
    cache = _nonempty_cache(tmp_path)
    monkeypatch.setattr(cli, "get_cache_dir", lambda: cache)

    def no_stdin(*_args: object, **_kwargs: object) -> NoReturn:
        raise EOFError

    monkeypatch.setattr(cli.Confirm, "ask", no_stdin)

    cli.cache_clean()

    assert (cache / "entry").exists()
