import re
from pathlib import Path

from loguru import logger


def _normalize_underscores(s: str) -> str:
    return re.sub(r"_+", "_", s)


def sanitize_net_name(name: str, with_suffix: bool = False) -> str:
    """Sanitize net name or path. If input is a path, only sanitize the
    basename. If input is a name, sanitize the whole string. Collapse
    multiple underscores.

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
