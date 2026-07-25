"""Host-side staging of user-provided inputs into the hidden cache.

The conversion runs inside a Docker container that only has the cache directory
bind-mounted (at ``/app/shared_with_container``). To let users reference input
files by *any* path on their machine, the launcher copies every file/directory
passed on the CLI into the cache (keyed by a content hash for de-duplication)
and rewrites the corresponding CLI token to the container-side cache path.

Files referenced *inside* a config file are not visible as CLI tokens, so when
a config file is passed we stage its whole containing directory; the container
then resolves the config's relative references against that copied directory
(see ``modelconverter.utils.filesystem_utils.set_input_base``).
"""

import hashlib
import shutil
from pathlib import Path

from loguru import logger

from modelconverter.utils.constants import get_cache_dir
from modelconverter.utils.filesystem_utils import get_protocol

# Container-side location of the cache mount (see docker_utils / constants).
_CONTAINER_SHARED_DIR = Path("/app/shared_with_container")

# CLI flags whose following value is a path that should be staged. Values that
# follow any other flag (e.g. --output-dir, --to) are left untouched.
_PATH_FLAGS = {"--path", "--config", "--model-path", "--input-path"}

# Target names are positional tokens that must not be mistaken for paths.
_TARGET_NAMES = {"rvc2", "rvc3", "rvc4", "hailo"}

# Extensions that clearly denote a file input.
_KNOWN_EXTS = {
    ".onnx",
    ".xml",
    ".bin",
    ".dlc",
    ".tflite",
    ".yaml",
    ".yml",
    ".json",
    ".pt",
    ".pth",
    ".tar",
    ".gz",
    ".zip",
    ".npy",
}

_CONFIG_EXTS = {".yaml", ".yml"}


def stage_inputs(tokens: list[str]) -> list[str]:
    """Copies path arguments among ``tokens`` into the cache and returns
    a new token list with those paths rewritten to their container-side
    locations.

    Non-path tokens, remote URLs and non-existent paths are left
    untouched.
    """
    inputs_dir = get_cache_dir() / "inputs"
    new_tokens: list[str] = []
    prev: str | None = None

    for token in tokens:
        staged = _maybe_stage_token(token, prev, inputs_dir)
        new_tokens.append(staged if staged is not None else token)
        prev = token

    return new_tokens


def _maybe_stage_token(
    token: str, prev: str | None, inputs_dir: Path
) -> str | None:
    """Returns the rewritten token if it should be staged, else
    ``None``."""
    # Handle the ``--flag=value`` form.
    if token.startswith("--") and "=" in token:
        flag, _, value = token.partition("=")
        if flag in _PATH_FLAGS:
            staged = _stage_value(value, inputs_dir)
            return f"{flag}={staged}" if staged is not None else None
        return None

    if token.startswith("--"):
        return None

    # A value explicitly introduced by a path flag.
    if prev in _PATH_FLAGS:
        return _stage_value(token, inputs_dir)

    # A value introduced by a flag we must not stage, or by an unknown/boolean
    # flag: leave it alone.
    if prev is not None and prev.startswith("--"):
        return None

    # A bare positional token (target, config-override value, ...). Only stage
    # it when it clearly looks like an existing local path.
    if _is_path_like(token):
        return _stage_value(token, inputs_dir)
    return None


def _is_path_like(value: str) -> bool:
    if value in _TARGET_NAMES:
        return False
    if get_protocol(value) != "file":
        return False
    p = Path(value).expanduser()
    if not p.exists():
        return False
    if p.is_dir():
        return True
    has_sep = "/" in value or "\\" in value
    return (
        p.is_absolute()
        or has_sep
        or value.startswith(("~", "."))
        or p.suffix.lower() in _KNOWN_EXTS
    )


def _stage_value(value: str, inputs_dir: Path) -> str | None:
    """Copies the file/dir at ``value`` into the cache and returns the
    container-side path, or ``None`` if it cannot/should not be
    staged."""
    if get_protocol(value) != "file":
        return None
    src = Path(value).expanduser()
    if not src.exists():
        return None
    src = src.resolve()

    if src.is_dir():
        dest = _stage_dir(src, inputs_dir)
        return _to_container(dest)

    # Config file: stage the whole containing directory so its relative
    # references resolve alongside it.
    if src.suffix.lower() in _CONFIG_EXTS:
        parent_dest = _stage_dir(src.parent, inputs_dir)
        return _to_container(parent_dest / src.name)

    # OpenVINO IR: stage the `.xml`/`.bin` pair together.
    if src.suffix.lower() in {".xml", ".bin"}:
        dest = _stage_ir_pair(src, inputs_dir)
        return _to_container(dest)

    dest = _stage_file(src, inputs_dir)
    return _to_container(dest)


def _stage_file(src: Path, inputs_dir: Path) -> Path:
    digest = _hash_file(src)
    dest = inputs_dir / digest / src.name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _log_copy(src, src.stat().st_size)
        shutil.copy2(src, dest)
    return dest


def _stage_ir_pair(src: Path, inputs_dir: Path) -> Path:
    xml = src.with_suffix(".xml")
    bin_ = src.with_suffix(".bin")
    members = [p for p in (xml, bin_) if p.exists()]
    digest = _hash_files(members)
    hash_dir = inputs_dir / digest
    for member in members:
        dest = hash_dir / member.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            _log_copy(member, member.stat().st_size)
            shutil.copy2(member, dest)
    return hash_dir / src.name


# Warn when a single staged directory is larger than this (e.g. a config file
# that happens to live in a big folder, whose whole parent gets copied).
_LARGE_DIR_BYTES = 1024**3  # 1 GiB


def _stage_dir(src: Path, inputs_dir: Path) -> Path:
    digest = _hash_dir(src)
    dest = inputs_dir / digest / src.name
    if not dest.exists():
        size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
        if size > _LARGE_DIR_BYTES:
            logger.warning(
                f"Caching a large directory {src} ({_human_size(size)}). "
                "Local files referenced by a config are resolved relative to "
                "the config file, so its whole directory is copied. Consider "
                "keeping configs in a dedicated folder. Run "
                "`modelconverter cache clean` to reclaim space."
            )
        _log_copy(src, size)
        shutil.copytree(src, dest)
    return dest


def _to_container(dest: Path) -> str:
    """Maps a host cache path to its container-side location."""
    relative = dest.relative_to(get_cache_dir())
    return str(_CONTAINER_SHARED_DIR / relative)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.name.encode())
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def _hash_dir(path: Path) -> str:
    """Fast content-fingerprint of a directory based on the relative
    paths, sizes and modification times of its files."""
    h = hashlib.sha256()
    h.update(path.name.encode())
    for f in sorted(path.rglob("*")):
        if f.is_file():
            stat = f.stat()
            rel = f.relative_to(path).as_posix()
            h.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _log_copy(src: Path, size: int) -> None:
    logger.info(f"Caching input {src} ({_human_size(size)})")


def _human_size(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PiB"
