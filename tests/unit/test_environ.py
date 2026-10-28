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
