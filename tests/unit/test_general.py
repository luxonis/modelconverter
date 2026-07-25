"""Unit tests for ``modelconverter.utils.general``."""

import pytest

from modelconverter.utils.general import (
    _normalize_underscores,
    sanitize_net_name,
)


class TestNormalizeUnderscores:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("a__b", "a_b"),
            ("a___b___c", "a_b_c"),
            ("a_b", "a_b"),
            ("ab", "ab"),
        ],
    )
    def test_collapse(self, value: str, expected: str):
        assert _normalize_underscores(value) == expected


class TestSanitizeNetName:
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
    def test_sanitize(self, name: str, with_suffix: bool, expected: str):
        assert sanitize_net_name(name, with_suffix=with_suffix) == expected

    def test_default_with_suffix_is_false(self):
        assert sanitize_net_name("foo@bar.onnx") == "foo_bar_onnx"
