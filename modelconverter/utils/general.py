"""General-purpose helpers shared across modelconverter.

Holds the small utilities that fit nowhere more specific: sanitizing
model names into something the conversion tools accept, and formatting
or parsing the byte sizes used to report on and bound the disk cache.
"""

import re
from pathlib import Path

from loguru import logger


def _normalize_underscores(s: str) -> str:
    return re.sub(r"_+", "_", s)


def sanitize_net_name(name: str, with_suffix: bool = False) -> str:
    """Sanitize net name or path.

    If input is a path, only sanitize the basename. If input is a name,
    sanitize the whole string. Collapse multiple underscores.

    Args:
        name: The name or path to sanitize.
        with_suffix: If ``True``, the suffix (file extension) is
            preserved and not sanitized.

    Returns:
        The sanitized name or path.

    Example:
        >>> sanitize_net_name("my model!.onnx")
        'my_model_onnx'
        >>> sanitize_net_name("my model!.onnx", with_suffix=True)
        'my_model_.onnx'
        >>> sanitize_net_name("a/b/my model!.onnx")
        'a/b/my_model_onnx'

    """
    p = Path(name)
    base, stem, suffix = p.name, p.stem, p.suffix

    if len(p.parts) > 1:
        if with_suffix and suffix:
            sanitized_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
            sanitized_stem = _normalize_underscores(sanitized_stem)
            sanitized_base = sanitized_stem + suffix
        else:
            sanitized_base = re.sub(r"[^a-zA-Z0-9_-]", "_", base)
            sanitized_base = _normalize_underscores(sanitized_base)
        if sanitized_base != p.name:
            logger.warning(
                f"Illegal characters detected in: '{p.name}'. Replacing with '_'. New name: '{sanitized_base}'"
            )
        return (
            str(p.parent / sanitized_base)
            if str(p.parent) != "."
            else sanitized_base
        )

    if with_suffix and suffix:
        sanitized_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
        sanitized_stem = _normalize_underscores(sanitized_stem)
        sanitized = sanitized_stem + suffix
    else:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        sanitized = _normalize_underscores(sanitized)
    if sanitized != name:
        logger.warning(
            f"Illegal characters detected in: '{name}'. Replacing with '_'. New name: '{sanitized}'"
        )
    return sanitized


def human_size(num: float) -> str:
    """Format a number of bytes for display.

    Args:
        num: Size in bytes.

    Returns:
        The size with one decimal place and a binary unit, such as
        ``"1.5 GiB"``.

    Example:
        >>> human_size(1536)
        '1.5 KiB'
        >>> human_size(5 * 1024**3)
        '5.0 GiB'

    """
    # The table stops where `parse_size` does: a size shown here is one a
    # user may hand back as a cache budget.
    for unit in ("B", "KiB", "MiB", "GiB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TiB"


def dir_stats(path: Path) -> tuple[int, int]:
    """Return the total size (bytes) and number of files under ``path``.

    Entries that cannot be stat'ed are skipped: a container run killed
    before its entrypoint could chown the mounts back leaves root-owned
    files behind, and neither reporting on the cache nor keeping it
    within its budget must be what breaks.
    """
    size = 0
    count = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                size += f.stat().st_size
                count += 1
        except OSError:
            continue
    return size, count


_SIZE_UNITS = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_SIZE_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)(?:I?B)?\s*$", re.IGNORECASE
)


def parse_size(value: str | int) -> int:
    """Parse a human-written byte size such as ``"50GiB"`` into bytes.

    The unit is optional and its prefixes are binary, matching
    `human_size` on the way out and Docker's own byte values on the way
    in: ``50G``, ``50GB`` and ``50GiB`` all mean the same 50 * 1024^3
    bytes, and a bare number is a count of bytes.

    Args:
        value: Size to parse. An integer is returned unchanged.

    Returns:
        The size in bytes.

    Raises:
        ValueError: If ``value`` is not a size.

    Example:
        >>> parse_size("50GiB")
        53687091200
        >>> parse_size("50GB")
        53687091200
        >>> parse_size("512m")
        536870912
        >>> parse_size(1024)
        1024

    """
    if isinstance(value, int):
        return value
    match = _SIZE_PATTERN.match(value)
    if match is None:
        raise ValueError(
            f"'{value}' is not a valid size. Expected a number with an "
            "optional unit -- 'b', 'k', 'm', 'g' or 't', spelled out as "
            "'kb'/'kib' if you like -- such as '512m' or '4GiB'. A bare "
            "number is a count of bytes."
        )
    number, unit = match.groups()
    return int(float(number) * _SIZE_UNITS[unit.upper() or "B"])
