"""Shared paths to bundled test assets."""

from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# A real photo, used as a conversion-precision inference input.
TEST_IMAGE = _DATA_DIR / "test_utils" / "test_image" / "orig.jpg"
