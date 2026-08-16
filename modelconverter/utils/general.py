import re
from pathlib import Path

from loguru import logger


def _normalize_underscores(s: str) -> str:
    return re.sub(r"_+", "_", s)


def sanitize_net_name(name: str, with_suffix: bool = False) -> str:
    """Sanitize net name or path.

    If input is a path, only sanitize the basename. If input is a name,
    sanitize the whole string. Collapse multiple underscores.

    @type name: str
    @param name: The name or path to sanitize.
    @type with_suffix: bool
    @param with_suffix: If True, the suffix (file extension) is
        preserved and not sanitized.
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
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PiB"


def dir_stats(path: Path) -> tuple[int, int]:
    """Returns the total size (bytes) and number of files under ``path``.

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
    """Parses a human-written byte size such as ``"50GiB"`` into bytes.

    The unit is optional and its prefixes are binary, matching
    L{human_size} on the way out and Docker's own byte values on the way
    in: ``50G``, ``50GB`` and ``50GiB`` all mean the same 50 * 1024^3
    bytes, and a bare number is a count of bytes.

    @raises ValueError: If C{value} is not a size.
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
