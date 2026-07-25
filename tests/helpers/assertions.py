"""Shared assertion helpers for the config unit tests."""

from pathlib import Path
from typing import Any

from luxonis_ml.typing import Params

from modelconverter.utils.config import Config

__all__ = ["load_and_compare"]


def load_and_compare(
    path: str | None,
    opts: list[str],
    expected: dict[str, Any],
    *,
    multistage: bool = False,
) -> None:
    """Loads a config (with CLI-style ``opts`` overrides) and asserts
    the resulting ``model_dump()`` equals ``expected``.

    For single-stage configs, ``expected`` is the stage dict; it is
    wrapped into the top-level ``{name, rich_logging, stages}`` shape
    using the input-model stem as the stage/config name.
    """
    overrides: Params = {opts[i]: opts[i + 1] for i in range(0, len(opts), 2)}
    config = Config.get_config(path, overrides).model_dump()
    if not multistage:
        name = Path(expected["input_model"]).stem
        expected = {
            "name": name,
            "rich_logging": True,
            "stages": {name: expected},
        }
    assert config == expected
