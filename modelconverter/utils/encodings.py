"""Parsing of externally supplied quantization encodings.

Quantization encodings describe the scale, offset and bit width used to
quantize activations and parameters of a model. They can be handed to
the RVC4 conversion to override the values derived from calibration.
The files come in several shapes, so this module normalizes them into
the ``Encodings`` model used by the conversion configuration.
"""

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from luxonis_ml.typing import ParamValue

if TYPE_CHECKING:
    from modelconverter.utils.config import Encodings


ALLOWED_ENCODING_KEYS = {
    "bitwidth",
    "is_symmetric",
    "dtype",
    "max",
    "min",
    "offset",
    "scale",
}


def _scalarize_encoding_value(value: ParamValue) -> ParamValue:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _normalize_encoding_item(
    item: Mapping[str, ParamValue],
) -> dict[str, ParamValue]:
    normalized = dict(item)

    if "bitwidth" not in normalized and "bw" in normalized:
        normalized["bitwidth"] = normalized["bw"]

    if "is_symmetric" not in normalized and "is_sym" in normalized:
        normalized["is_symmetric"] = normalized["is_sym"]

    dtype = normalized.get("dtype")
    if isinstance(dtype, str):
        normalized["dtype"] = dtype.lower()

    for key in ["scale", "offset", "min", "max"]:
        if key in normalized:
            normalized[key] = _scalarize_encoding_value(normalized[key])

    return {
        key: value
        for key, value in normalized.items()
        if key in ALLOWED_ENCODING_KEYS
    }


def _expand_encoding_item(
    item: Mapping[str, ParamValue],
) -> list[dict[str, ParamValue]]:
    normalized = _normalize_encoding_item(item)
    vector_lengths = [
        len(value)
        for key, value in normalized.items()
        if key in {"scale", "offset", "min", "max"} and isinstance(value, list)
    ]
    if not vector_lengths:
        return [normalized]

    size = vector_lengths[0]
    if any(length != size for length in vector_lengths):
        raise ValueError(
            "Per-channel encoding fields must have matching lengths."
        )

    expanded = []
    for idx in range(size):
        entry = {}
        for key, value in normalized.items():
            entry[key] = value[idx] if isinstance(value, list) else value
        expanded.append(entry)
    return expanded


def _normalize_encoding_group(
    entries: ParamValue,
) -> dict[str, list[dict[str, ParamValue]]]:
    if isinstance(entries, dict):
        normalized = {}
        for name, value in entries.items():
            values = value if isinstance(value, list) else [value]
            items = []
            for item in values:
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Expected dict encoding entry, got {type(item).__name__}."
                    )
                items.extend(_expand_encoding_item(item))
            normalized[name] = items
        return normalized

    if not isinstance(entries, list):
        raise TypeError(
            f"Expected encoding group to be a list or dict, got {type(entries).__name__}."
        )

    normalized = {}
    for item in entries:
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected dict encoding entry, got {type(item).__name__}."
            )
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Missing or invalid tensor name in entry: {item}"
            )
        normalized.setdefault(name, []).extend(_expand_encoding_item(item))
    return normalized


def parse_encodings(value: "ParamValue | Encodings") -> "Encodings":
    """Parse quantization encodings into an ``Encodings`` model.

    Accepts an ``Encodings`` instance, which is returned unchanged, a
    JSON string, or a dictionary with ``activation_encodings`` and
    ``param_encodings`` keys. Each of the two groups may either be a
    mapping from tensor name to entries, or a list of entries carrying
    a ``name`` field. Entries are normalized on the way: the legacy
    ``bw`` and ``is_sym`` keys are renamed to ``bitwidth`` and
    ``is_symmetric``, the ``dtype`` is lower-cased, unknown keys are
    dropped, single-element lists are unwrapped and per-channel entries
    are expanded into one entry per channel.

    Args:
        value: Encodings to parse.

    Returns:
        Parsed encodings.

    Raises:
        TypeError: If ``value`` is not a dictionary once deserialized,
            if a group is neither a list nor a dictionary, or if an
            individual entry is not a dictionary.
        ValueError: If an entry in a list-shaped group has a missing or
            invalid tensor name, or if the per-channel fields of an
            entry have different lengths.

    """
    from modelconverter.utils.config import Encodings

    if isinstance(value, Encodings):
        return value

    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected encodings to deserialize to a dict, got {type(value).__name__}."
        )

    return Encodings.model_validate(
        {
            "activation_encodings": _normalize_encoding_group(
                value.get("activation_encodings", {})
            ),
            "param_encodings": _normalize_encoding_group(
                value.get("param_encodings", {})
            ),
        }
    )
