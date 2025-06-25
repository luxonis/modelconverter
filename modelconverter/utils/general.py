import re

from loguru import logger


def sanitize_net_name(name: str) -> str:
    """Sanitize net name since only alphanumeric chars, hyphens and
    underscores are allowed."""
    if re.search(r"[^a-zA-Z0-9_-]", name):
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        logger.warning(
            f"Illegal characters detected in: {name}. Replacing with '_'. New name: {sanitized}"
        )
        return sanitized
    return name
