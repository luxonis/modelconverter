"""Tests for the keyring fallback of the HubAI API key."""

import importlib
from pathlib import Path

import pytest

from modelconverter.utils.environ import Environ

# `modelconverter.utils.environ` as an attribute is the settings instance,
# so the module itself has to come from the import system.
environ_module = importlib.import_module("modelconverter.utils.environ")


def _failing_lookup(service_name: str, username: str) -> str | None:
    """Fail like a keyring without a usable backend."""
    raise RuntimeError(f"no keyring entry {service_name}/{username}")


def test_failed_keyring_lookup_keeps_an_empty_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HUBAI_API_KEY", "")
    monkeypatch.setattr(
        environ_module, "get_password_with_timeout", _failing_lookup
    )

    api_key = Environ().HUBAI_API_KEY

    assert api_key == ""


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("return f'{service_name}/{username}'", "test-service/test-user"),
        ("return None", None),
        ("raise RuntimeError('backend unavailable')", None),
    ],
)
def test_keyring_lookup_in_fresh_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    expected: str | None,
):
    backend = tmp_path / "keyring.py"
    backend.write_text(
        f"def get_password(service_name, username):\n    {body}\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    assert (
        environ_module.get_password_with_timeout("test-service", "test-user")
        == expected
    )


def test_keyring_lookup_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "keyring.py").write_text(
        "import time\n"
        "def get_password(service_name, username):\n"
        "    time.sleep(10)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    assert (
        environ_module.get_password_with_timeout("test", "user", timeout=0.1)
        is None
    )


def test_keyring_lookup_after_threaded_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Python 3.12's fork warning must not deadlock with Loguru's fork locks."""
    import os
    import subprocess
    import sys

    (tmp_path / "keyring.py").write_text(
        "def get_password(service_name, username):\n    return 'test-password'\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    # Skip the settings singleton's lookup so this probe controls exactly when
    # logging and another thread exist. No real credential backend is used.
    monkeypatch.setenv("HUBAI_API_KEY", "test-key")
    script = """
import threading
import warnings
with warnings.catch_warnings():
    from luxonis_ml.utils import setup_logging
from modelconverter.utils.environ import get_password_with_timeout

setup_logging()
warnings.simplefilter("always", DeprecationWarning)
stop = threading.Event()
thread = threading.Thread(target=stop.wait, daemon=True)
thread.start()
try:
    assert get_password_with_timeout("test", "user", timeout=3) == "test-password"
finally:
    stop.set()
    thread.join()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
