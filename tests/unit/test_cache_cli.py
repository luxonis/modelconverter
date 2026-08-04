from pathlib import Path
from typing import NoReturn

import pytest

from modelconverter import __main__ as cli


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
