"""Unit tests for ``modelconverter.utils.general``."""

import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from modelconverter.utils.general import (
    _normalize_underscores,
    sanitize_net_name,
)

# Printable ASCII except "/" (the two ranges skip 0x2F), minus the two
# components ``PurePath`` folds away -- "." and ".." are a question
# about paths, not about naming.
_names = st.from_regex(r"[ -.0-~]{1,12}", fullmatch=True).filter(
    lambda part: part not in {".", ".."}
)
_paths = st.lists(_names, min_size=1, max_size=3).map("/".join)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a__b", "a_b"),
        ("a___b___c", "a_b_c"),
        ("a_b", "a_b"),
        ("ab", "ab"),
    ],
)
def test_collapse(value: str, expected: str):
    assert _normalize_underscores(value) == expected


@pytest.mark.parametrize(
    ("name", "with_suffix", "expected"),
    [
        # --- Plain name, no path (len(parts) == 1) ---
        # with_suffix=False sanitizes the whole string, so the dot goes too.
        ("model.onnx", False, "model_onnx"),
        ("model.onnx", True, "model.onnx"),
        ("foo@bar.onnx", False, "foo_bar_onnx"),
        ("foo@bar.onnx", True, "foo_bar.onnx"),
        ("foo bar.onnx", False, "foo_bar_onnx"),
        # Underscore collapsing on the non-path branch.
        ("foo@@bar", False, "foo_bar"),
        ("a@@b.onnx", True, "a_b.onnx"),
        # with_suffix=True but no extension -> else branch.
        ("noext", True, "noext"),
        ("no@ext", True, "no_ext"),
        # Relative "./..." collapses to a bare name (parts == 1), but the
        # non-path branch sanitizes the raw string, so the leading "./"
        # becomes a single underscore.
        ("./foo@bar.onnx", False, "_foo_bar_onnx"),
        ("./foo@bar.onnx", True, "foo_bar.onnx"),
        # --- Name with a parent path (len(parts) > 1) ---
        ("dir/model.onnx", False, "dir/model_onnx"),
        ("dir/model.onnx", True, "dir/model.onnx"),
        ("dir/foo@bar.onnx", False, "dir/foo_bar_onnx"),
        ("dir/foo@bar.onnx", True, "dir/foo_bar.onnx"),
        ("a/b/foo bar.onnx", False, "a/b/foo_bar_onnx"),
        # Underscore collapsing on the path branch.
        ("dir/a@@b.onnx", True, "dir/a_b.onnx"),
        # with_suffix=True but no extension in basename -> else.
        ("dir/noext", True, "dir/noext"),
        ("dir/no@ext", True, "dir/no_ext"),
    ],
)
def test_sanitize(name: str, with_suffix: bool, expected: str):
    assert sanitize_net_name(name, with_suffix=with_suffix) == expected


def test_default_with_suffix_is_false():
    assert sanitize_net_name("foo@bar.onnx") == "foo_bar_onnx"


# --- Properties -----------------------------------------------------
#
# Sanitized names end up as filenames inside the conversion containers
# and as node names in the produced archives, so the guarantees below
# have to hold for every name a user can type, not just the shapes the
# cases above enumerate.


@given(value=st.text(alphabet="_ab", max_size=12))
def test_underscore_runs_always_collapse(value: str):
    collapsed = _normalize_underscores(value)
    assert "__" not in collapsed
    assert _normalize_underscores(collapsed) == collapsed


@given(name=_paths, with_suffix=st.booleans())
def test_sanitizing_is_idempotent(name: str, with_suffix: bool):
    once = sanitize_net_name(name, with_suffix=with_suffix)
    assert sanitize_net_name(once, with_suffix=with_suffix) == once


@given(name=_paths)
def test_sanitized_basename_is_a_safe_identifier(name: str):
    """Nothing but ``[A-Za-z0-9_-]`` survives in the basename."""
    sanitized = sanitize_net_name(name)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", Path(sanitized).name)


@given(name=_paths)
def test_with_suffix_preserves_the_extension_verbatim(name: str):
    """``with_suffix=True`` sanitizes the stem and nothing else.

    Conversion dispatches on the extension, so rewriting it would send
    a model down the wrong backend.
    """
    sanitized = sanitize_net_name(name, with_suffix=True)
    assert Path(sanitized).suffix == Path(name).suffix
    assert re.fullmatch(r"[A-Za-z0-9_-]*", Path(sanitized).stem)


@given(name=_paths, with_suffix=st.booleans())
def test_only_the_basename_is_rewritten(name: str, with_suffix: bool):
    """A model kept in a directory stays in that directory."""
    sanitized = sanitize_net_name(name, with_suffix=with_suffix)
    assert Path(sanitized).parent == Path(name).parent
