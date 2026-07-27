"""Host-side unit tests for ``modelconverter.utils.environ``.

The module exposes `get_password_with_timeout` (a keyring lookup
guarded by a subprocess timeout) and the `Environ` pydantic-
settings model whose ``after`` validator falls back to keyring for
``HUBAI_API_KEY``. Both are fully reachable host-side; the real keyring
backend is monkeypatched so no OS credential store is touched.
"""

import multiprocessing
import sys
import time

import pytest

from modelconverter.utils.environ import (
    Environ,
    get_password_with_timeout,
)

# ``modelconverter.utils`` re-exports the ``environ`` *instance*, which
# shadows the submodule under attribute access; fetch the real module
# object from ``sys.modules`` so ``keyring`` / ``get_password_with_timeout``
# can be monkeypatched on it.
env_mod = sys.modules["modelconverter.utils.environ"]


@pytest.mark.skipif(
    multiprocessing.get_start_method(allow_none=True) != "fork",
    reason="keyring subprocess test only works on fork() platforms",
)
def test_returns_password(monkeypatch: pytest.MonkeyPatch):
    """A fast keyring hit is forwarded back through the queue."""
    monkeypatch.setattr(
        env_mod.keyring, "get_password", lambda service, user: "secret"
    )
    assert get_password_with_timeout("ModelConverter", "api_key") == "secret"


def test_missing_password_returns_none(monkeypatch: pytest.MonkeyPatch):
    """A ``None`` from keyring propagates as ``None``."""
    monkeypatch.setattr(
        env_mod.keyring, "get_password", lambda service, user: None
    )
    assert get_password_with_timeout("ModelConverter", "api_key") is None


def test_keyring_exception_returns_none(monkeypatch: pytest.MonkeyPatch):
    """An exception inside the child process yields ``None``."""

    def boom(service: str, user: str) -> str:  # pragma: no cover
        raise RuntimeError("no backend")

    monkeypatch.setattr(env_mod.keyring, "get_password", boom)
    assert get_password_with_timeout("ModelConverter", "api_key") is None


def test_timeout_returns_none(monkeypatch: pytest.MonkeyPatch):
    """A hanging keyring backend is terminated and returns ``None``."""

    def slow(service: str, user: str) -> str:  # pragma: no cover
        time.sleep(30)
        return "never"

    monkeypatch.setattr(env_mod.keyring, "get_password", slow)
    start = time.monotonic()
    result = get_password_with_timeout(
        "ModelConverter", "api_key", timeout=0.2
    )
    assert result is None
    # Must have honoured the short timeout, not the 30s sleep.
    assert time.monotonic() - start < 5


def test_explicit_api_key_kept(monkeypatch: pytest.MonkeyPatch):
    """An explicitly provided key short-circuits the keyring lookup."""

    def fail(*args, **kwargs) -> str:
        raise AssertionError("keyring must not be consulted")

    monkeypatch.setattr(env_mod, "get_password_with_timeout", fail)
    env = Environ(HUBAI_API_KEY="explicit-key")
    assert env.HUBAI_API_KEY == "explicit-key"


def test_keyring_fallback_used(monkeypatch: pytest.MonkeyPatch):
    """With no key set, the keyring value is adopted."""
    monkeypatch.delenv("HUBAI_API_KEY", raising=False)
    monkeypatch.setattr(
        env_mod,
        "get_password_with_timeout",
        lambda *args, **kwargs: "from-keyring",
    )
    env = Environ()
    assert env.HUBAI_API_KEY == "from-keyring"


def test_keyring_returns_none(monkeypatch: pytest.MonkeyPatch):
    """A ``None`` keyring result leaves the key unset."""
    monkeypatch.delenv("HUBAI_API_KEY", raising=False)
    monkeypatch.setattr(
        env_mod,
        "get_password_with_timeout",
        lambda *args, **kwargs: None,
    )
    env = Environ()
    assert env.HUBAI_API_KEY is None


def test_keyring_exception_suppressed(monkeypatch: pytest.MonkeyPatch):
    """An exception in the fallback is swallowed by ``suppress``."""
    monkeypatch.delenv("HUBAI_API_KEY", raising=False)

    def boom(*args, **kwargs) -> str:
        raise RuntimeError("keyring exploded")

    monkeypatch.setattr(env_mod, "get_password_with_timeout", boom)
    env = Environ()
    assert env.HUBAI_API_KEY is None


def test_default_url(monkeypatch: pytest.MonkeyPatch):
    """The default HubAI URL is exposed when the env var is absent."""
    monkeypatch.delenv("HUBAI_URL", raising=False)
    monkeypatch.setattr(
        env_mod,
        "get_password_with_timeout",
        lambda *args, **kwargs: None,
    )
    env = Environ()
    assert env.HUBAI_URL == "https://easyml.cloud.luxonis.com/"
