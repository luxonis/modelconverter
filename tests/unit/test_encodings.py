"""Unit tests for ``modelconverter.utils.encodings``."""

import json

import pytest

from modelconverter.utils.config import Encodings
from modelconverter.utils.encodings import (
    _expand_encoding_item,
    _normalize_encoding_group,
    _normalize_encoding_item,
    _scalarize_encoding_value,
    parse_encodings,
)


class TestScalarizeEncodingValue:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([0.5], 0.5),
            ([0.5, 0.6], [0.5, 0.6]),
            (0.5, 0.5),
            ([], []),
        ],
    )
    def test_scalarize(
        self, value: float | list[float], expected: float | list[float]
    ):
        assert _scalarize_encoding_value(value) == expected


class TestNormalizeEncodingItem:
    def test_bw_alias(self):
        assert _normalize_encoding_item({"bw": 8})["bitwidth"] == 8

    def test_bw_alias_not_overriding_existing(self):
        item = {"bitwidth": 16, "bw": 8}
        assert _normalize_encoding_item(item)["bitwidth"] == 16

    def test_is_sym_alias(self):
        item = {"is_sym": True}
        assert _normalize_encoding_item(item)["is_symmetric"] is True

    def test_is_sym_alias_not_overriding_existing(self):
        item = {"is_symmetric": False, "is_sym": True}
        assert _normalize_encoding_item(item)["is_symmetric"] is False

    def test_dtype_lowercased(self):
        assert _normalize_encoding_item({"dtype": "INT"})["dtype"] == "int"

    def test_dtype_non_string_untouched(self):
        # ``dtype`` absent -> ``.get`` returns None -> not str branch.
        assert "dtype" not in _normalize_encoding_item({"scale": 1.0})

    def test_scalarize_single_element_vectors(self):
        item = {
            "scale": [0.1],
            "offset": [2],
            "min": [-1.0],
            "max": [1.0],
        }
        normalized = _normalize_encoding_item(item)
        assert normalized == {
            "scale": 0.1,
            "offset": 2,
            "min": -1.0,
            "max": 1.0,
        }

    def test_multi_element_vectors_preserved(self):
        item = {"scale": [0.1, 0.2]}
        assert _normalize_encoding_item(item)["scale"] == [0.1, 0.2]

    def test_disallowed_keys_dropped(self):
        item = {"scale": 1.0, "name": "foo", "unknown": 123}
        assert _normalize_encoding_item(item) == {"scale": 1.0}


class TestExpandEncodingItem:
    def test_no_vector_fields_single_item(self):
        item = {"scale": 0.1, "offset": 0}
        assert _expand_encoding_item(item) == [{"scale": 0.1, "offset": 0}]

    def test_per_channel_expansion(self):
        item = {"scale": [0.1, 0.2], "offset": [1, 2], "bitwidth": 8}
        expanded = _expand_encoding_item(item)
        assert expanded == [
            {"scale": 0.1, "offset": 1, "bitwidth": 8},
            {"scale": 0.2, "offset": 2, "bitwidth": 8},
        ]

    def test_mismatched_lengths_raise(self):
        # Both must stay vectors (len != 1, else scalarized) and differ.
        item = {"scale": [0.1, 0.2], "min": [1.0, 2.0, 3.0]}
        with pytest.raises(ValueError, match="matching lengths"):
            _expand_encoding_item(item)


class TestNormalizeEncodingGroup:
    def test_dict_form_scalar_value(self):
        group = {"tensor": {"scale": 0.1, "bitwidth": 8}}
        result = _normalize_encoding_group(group)
        assert result == {"tensor": [{"scale": 0.1, "bitwidth": 8}]}

    def test_dict_form_list_value(self):
        group = {"tensor": [{"scale": 0.1}, {"scale": 0.2}]}
        result = _normalize_encoding_group(group)
        assert result == {"tensor": [{"scale": 0.1}, {"scale": 0.2}]}

    def test_dict_form_non_dict_entry_raises(self):
        with pytest.raises(TypeError, match="dict encoding entry"):
            _normalize_encoding_group({"tensor": [123]})

    def test_list_form_name_from_item(self):
        group = [{"name": "tensor", "scale": 0.1}]
        result = _normalize_encoding_group(group)
        assert result == {"tensor": [{"scale": 0.1}]}

    def test_list_form_accumulates_same_name(self):
        group = [
            {"name": "tensor", "scale": 0.1},
            {"name": "tensor", "scale": 0.2},
        ]
        result = _normalize_encoding_group(group)
        assert result == {"tensor": [{"scale": 0.1}, {"scale": 0.2}]}

    @pytest.mark.parametrize("bad_name", [{}, {"name": ""}, {"name": 5}])
    def test_list_form_missing_or_invalid_name_raises(self, bad_name: dict):
        with pytest.raises(ValueError, match="tensor name"):
            _normalize_encoding_group([bad_name])

    def test_list_form_non_dict_entry_raises(self):
        with pytest.raises(TypeError, match="dict encoding entry"):
            _normalize_encoding_group(["not-a-dict"])

    def test_neither_list_nor_dict_raises(self):
        with pytest.raises(TypeError, match="list or dict"):
            _normalize_encoding_group(42)


class TestParseEncodings:
    def test_encodings_instance_passthrough(self):
        enc = Encodings(activation_encodings={}, param_encodings={})
        assert parse_encodings(enc) is enc

    def test_json_string_input(self):
        raw = json.dumps(
            {
                "activation_encodings": {
                    "act": {"scale": 0.1, "bw": 8, "dtype": "INT"}
                },
                "param_encodings": [
                    {"name": "weight", "scale": [0.1, 0.2]}
                ],
            }
        )
        enc = parse_encodings(raw)
        assert isinstance(enc, Encodings)
        act = enc.activation_encodings["act"][0]
        assert act.scale == 0.1
        assert act.bitwidth == 8
        assert act.dtype == "int"
        assert len(enc.param_encodings["weight"]) == 2

    def test_dict_input(self):
        enc = parse_encodings(
            {"activation_encodings": {"act": {"scale": 0.5}}}
        )
        assert isinstance(enc, Encodings)
        assert enc.activation_encodings["act"][0].scale == 0.5
        assert enc.param_encodings == {}

    def test_list_json_raises_type_error(self):
        with pytest.raises(TypeError, match="deserialize to a dict"):
            parse_encodings("[]")

    def test_raw_int_raises_type_error(self):
        with pytest.raises(TypeError, match="deserialize to a dict"):
            parse_encodings(5)
