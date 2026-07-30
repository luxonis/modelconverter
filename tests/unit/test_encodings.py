"""Unit tests for ``modelconverter.utils.encodings``.

Quantization encodings arrive from AIMET in several shapes -- keyed by tensor name
or as a flat list, per-tensor or per-channel, with the short ``bw``/``is_sym``
aliases or the long spellings -- and all have to collapse into the same
``Encodings`` model. The cases below pin the individual normalisation rules; the
properties at the bottom assert the invariants that hold across combinations.
"""

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from modelconverter.utils.config import Encodings
from modelconverter.utils.encodings import (
    ALLOWED_ENCODING_KEYS,
    _expand_encoding_item,
    _normalize_encoding_group,
    _normalize_encoding_item,
    _scalarize_encoding_value,
    parse_encodings,
)
from tests.helpers.strategies import (
    VECTOR_ENCODING_KEYS,
    encoding_groups,
    encoding_items,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([0.5], 0.5),
        ([0.5, 0.6], [0.5, 0.6]),
        (0.5, 0.5),
        ([], []),
    ],
)
def test_scalarize(value: float | list[float], expected: float | list[float]):
    assert _scalarize_encoding_value(value) == expected


def test_bw_alias():
    assert _normalize_encoding_item({"bw": 8})["bitwidth"] == 8


def test_bw_alias_not_overriding_existing():
    item = {"bitwidth": 16, "bw": 8}
    assert _normalize_encoding_item(item)["bitwidth"] == 16


def test_is_sym_alias():
    item = {"is_sym": True}
    assert _normalize_encoding_item(item)["is_symmetric"] is True


def test_is_sym_alias_not_overriding_existing():
    item = {"is_symmetric": False, "is_sym": True}
    assert _normalize_encoding_item(item)["is_symmetric"] is False


def test_dtype_lowercased():
    assert _normalize_encoding_item({"dtype": "INT"})["dtype"] == "int"


def test_dtype_non_string_untouched():
    # ``dtype`` absent -> ``.get`` returns None -> not str branch.
    assert "dtype" not in _normalize_encoding_item({"scale": 1.0})


def test_scalarize_single_element_vectors():
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


def test_multi_element_vectors_preserved():
    item = {"scale": [0.1, 0.2]}
    assert _normalize_encoding_item(item)["scale"] == [0.1, 0.2]


def test_disallowed_keys_dropped():
    item = {"scale": 1.0, "name": "foo", "unknown": 123}
    assert _normalize_encoding_item(item) == {"scale": 1.0}


def test_no_vector_fields_single_item():
    item = {"scale": 0.1, "offset": 0}
    assert _expand_encoding_item(item) == [{"scale": 0.1, "offset": 0}]


def test_per_channel_expansion():
    item = {"scale": [0.1, 0.2], "offset": [1, 2], "bitwidth": 8}
    expanded = _expand_encoding_item(item)
    assert expanded == [
        {"scale": 0.1, "offset": 1, "bitwidth": 8},
        {"scale": 0.2, "offset": 2, "bitwidth": 8},
    ]


def test_mismatched_lengths_raise():
    # Both must stay vectors (len != 1, else scalarized) and differ.
    item = {"scale": [0.1, 0.2], "min": [1.0, 2.0, 3.0]}
    with pytest.raises(ValueError, match="matching lengths"):
        _expand_encoding_item(item)


def test_dict_form_scalar_value():
    group = {"tensor": {"scale": 0.1, "bitwidth": 8}}
    normalized = _normalize_encoding_group(group)
    assert normalized == {"tensor": [{"scale": 0.1, "bitwidth": 8}]}


def test_dict_form_list_value():
    group = {"tensor": [{"scale": 0.1}, {"scale": 0.2}]}
    normalized = _normalize_encoding_group(group)
    assert normalized == {"tensor": [{"scale": 0.1}, {"scale": 0.2}]}


def test_dict_form_non_dict_entry_raises():
    with pytest.raises(TypeError, match="dict encoding entry"):
        _normalize_encoding_group({"tensor": [123]})


def test_list_form_name_from_item():
    group = [{"name": "tensor", "scale": 0.1}]
    normalized = _normalize_encoding_group(group)
    assert normalized == {"tensor": [{"scale": 0.1}]}


def test_list_form_accumulates_same_name():
    group = [
        {"name": "tensor", "scale": 0.1},
        {"name": "tensor", "scale": 0.2},
    ]
    normalized = _normalize_encoding_group(group)
    assert normalized == {"tensor": [{"scale": 0.1}, {"scale": 0.2}]}


@pytest.mark.parametrize("bad_name", [{}, {"name": ""}, {"name": 5}])
def test_list_form_missing_or_invalid_name_raises(bad_name: dict):
    with pytest.raises(ValueError, match="tensor name"):
        _normalize_encoding_group([bad_name])


def test_list_form_non_dict_entry_raises():
    with pytest.raises(TypeError, match="dict encoding entry"):
        _normalize_encoding_group(["not-a-dict"])


def test_neither_list_nor_dict_raises():
    with pytest.raises(TypeError, match="list or dict"):
        _normalize_encoding_group(42)


def test_encodings_instance_passthrough():
    enc = Encodings(activation_encodings={}, param_encodings={})
    assert parse_encodings(enc) is enc


def test_json_string_input():
    raw = json.dumps(
        {
            "activation_encodings": {
                "act": {"scale": 0.1, "bw": 8, "dtype": "INT"}
            },
            "param_encodings": [{"name": "weight", "scale": [0.1, 0.2]}],
        }
    )
    enc = parse_encodings(raw)
    assert isinstance(enc, Encodings)
    act = enc.activation_encodings["act"][0]
    assert act.scale == 0.1
    assert act.bitwidth == 8
    assert act.dtype == "int"
    assert len(enc.param_encodings["weight"]) == 2


def test_dict_input():
    enc = parse_encodings({"activation_encodings": {"act": {"scale": 0.5}}})
    assert isinstance(enc, Encodings)
    assert enc.activation_encodings["act"][0].scale == 0.5
    assert enc.param_encodings == {}


def test_list_json_raises_type_error():
    with pytest.raises(TypeError, match="deserialize to a dict"):
        parse_encodings("[]")


def test_raw_int_raises_type_error():
    with pytest.raises(TypeError, match="deserialize to a dict"):
        parse_encodings(5)


@given(item=encoding_items())
def test_normalization_keeps_only_known_keys(item: dict[str, Any]):
    """Aliases and vendor extras never reach the pydantic model.

    ``QuantizationOverridesItem`` forbids extra fields, so anything the
    normaliser fails to drop becomes a validation error much later, at
    config load time.
    """
    normalized = _normalize_encoding_item(item)
    assert set(normalized) <= ALLOWED_ENCODING_KEYS


@given(item=encoding_items())
def test_normalization_is_idempotent(item: dict[str, Any]):
    once = _normalize_encoding_item(item)
    assert _normalize_encoding_item(once) == once


@given(item=encoding_items())
def test_aliases_resolve_to_their_long_spelling(item: dict[str, Any]):
    normalized = _normalize_encoding_item(item)
    for alias, key in [("bw", "bitwidth"), ("is_sym", "is_symmetric")]:
        if key in item:
            # An explicit long spelling always wins over the alias.
            assert normalized[key] == item[key]
        elif alias in item:
            assert normalized[key] == item[alias]
        else:
            assert key not in normalized


@given(
    length=st.integers(min_value=1, max_value=6),
    data=st.data(),
)
def test_expansion_yields_one_scalar_entry_per_channel(
    length: int, data: st.DataObject
):
    """Per-channel vectors fan out into one entry per channel.

    Every entry keeps the full key set -- the scalar fields (``bitwidth``,
    ``dtype``, ...) are broadcast rather than dropped -- and no list
    survives the expansion.
    """
    item = data.draw(encoding_items(vector_length=length))
    normalized = _normalize_encoding_item(item)
    expanded = _expand_encoding_item(item)

    # A one-element vector is scalarized first, leaving a single entry.
    is_per_channel = any(
        isinstance(normalized.get(key), list) for key in VECTOR_ENCODING_KEYS
    )
    assert len(expanded) == (length if is_per_channel else 1)
    for i, entry in enumerate(expanded):
        assert set(entry) == set(normalized)
        assert not any(isinstance(value, list) for value in entry.values())
        for key, value in normalized.items():
            expected = value[i] if isinstance(value, list) else value
            assert entry[key] == expected


@given(group=encoding_groups())
def test_list_and_dict_group_forms_agree(
    group: dict[str, list[dict[str, Any]]],
):
    """The two wire formats are the same data, so they normalize alike.

    A group may arrive keyed by tensor name or as a flat list carrying
    ``name`` on each entry; nothing downstream should be able to tell
    which one was used.
    """
    # ``name`` last: the list form carries the tensor name on the entry
    # itself, so it overrides any stray ``name`` the entry already had.
    as_list = [
        {**item, "name": name}
        for name, items in group.items()
        for item in items
    ]
    assert _normalize_encoding_group(as_list) == _normalize_encoding_group(
        group
    )


@given(activations=encoding_groups(), params=encoding_groups())
def test_parse_encodings_round_trips_through_json(
    activations: dict[str, list[dict[str, Any]]],
    params: dict[str, list[dict[str, Any]]],
):
    """A JSON payload and the dict it decodes to give the same model.

    Users pass encodings either inline in the YAML config or as a path
    to a ``.json`` file, and the two paths must not diverge.
    """
    payload = {
        "activation_encodings": activations,
        "param_encodings": params,
    }
    assert parse_encodings(json.dumps(payload)) == parse_encodings(payload)


@given(activations=encoding_groups())
def test_parsed_encodings_are_idempotent(
    activations: dict[str, list[dict[str, Any]]],
):
    """An already-parsed ``Encodings`` passes straight through."""
    parsed = parse_encodings({"activation_encodings": activations})
    assert parse_encodings(parsed) is parsed
