"""Resolution of the local and remote paths modelconverter reads.

Models, configs and calibration data may be given either as local paths
or as URLs into bucket storage. The helpers here download the remote
ones, upload results back, and resolve relative local paths against the
directory of the active config file and the shared root (the cache
directory inside the container, the working directory on the host).
"""

import os
from contextvars import ContextVar
from pathlib import Path

from luxonis_ml.typing import PathType
from luxonis_ml.utils import LuxonisFileSystem

from modelconverter.utils.constants import SHARED_DIR, in_docker

# Base directory against which relative local paths are resolved when they do
# not exist as-is. It is set to the directory of the active config file while
# it is being parsed (see `set_input_base`), so that files referenced *inside*
# a config (calibration data, scripts, encodings, ...) are resolved relative to
# the config file. It is tried before -- not instead of -- the default root:
# the cache directory (inside the container) or the current working directory
# (native runs). See `get_input_bases`.
_input_base: ContextVar[Path | None] = ContextVar("input_base", default=None)


def set_input_base(base: Path | None) -> None:
    """Set the base directory for resolving relative local paths.

    Pass ``None`` to reset to the default.
    """
    _input_base.set(base)


def get_input_bases() -> list[Path]:
    """Return the base directories tried, in order, for a relative
    local path that does not exist as-is.

    The active config's directory comes first, then the default root. The
    default has to stay as a fallback: a config downloaded from remote
    storage lands in the cache, so the paths it references relative to the
    shared root would otherwise stop resolving.
    """
    default = SHARED_DIR if in_docker() else Path.cwd()
    base = _input_base.get()
    if base is None or base == default:
        return [default]
    return [base, default]


def resolve_input_path(string: str) -> Path | None:
    """Resolve a relative local path against the input bases.

    Returns ``None`` if the path exists under none of them.
    """
    for base in get_input_bases():
        candidate = base / string
        if candidate.exists():
            return candidate
    return None


def resolve_path(string: str, dest: Path) -> Path:
    """Download the file from remote or return the path otherwise."""
    protocol = get_protocol(string)
    if protocol != "file":
        path = download_from_remote(string, dest)
    else:
        path = Path(string).expanduser()
    if not path.exists():
        resolved = resolve_input_path(string)
        if resolved is None:
            raise ValueError(f"Path `{string}` does not exist.")
        path = resolved
    return path


def download_from_remote(
    url: str, dest: PathType, max_files: int = -1
) -> Path:
    """Download file(s) from remote bucket storage.

    It could be a single file, an entire directory, or ``max_files``
    within a directory.

    Args:
        url: URL of the file or directory to download.
        dest: Local directory to download into.
        max_files: Maximum number of files to download from a
            directory. Negative means no limit.

    Returns:
        Path to the downloaded file or directory.

    """
    absolute_path, remote_path = LuxonisFileSystem.split_full_path(url)
    if isinstance(dest, str):
        dest = Path(dest)
    local_path = dest / remote_path
    fs = LuxonisFileSystem(absolute_path)

    if fs.is_directory(remote_path):
        for i, remote_file in enumerate(fs.walk_dir(remote_path)):
            if i == max_files:
                break
            if not local_path.exists() or not os.getenv(
                "MODELCONVERTER_UNSAFE_CACHE"
            ):
                fs.get_file(
                    remote_file, str(local_path / Path(remote_file).name)
                )

    elif not local_path.exists() or not os.getenv(
        "MODELCONVERTER_UNSAFE_CACHE"
    ):
        fs.get_file(remote_path, str(local_path))

    return local_path


def upload_to_remote(
    local_path: PathType,
    url: str,
    put_file_plugin: str | None = None,
) -> None:
    """Upload a file or a directory to remote bucket storage."""
    local_path = Path(local_path)
    absolute_path, remote_path = LuxonisFileSystem.split_full_path(url)
    fs = LuxonisFileSystem(absolute_path, put_file_plugin=put_file_plugin)

    if local_path.is_dir():
        fs.put_dir(local_path, remote_path)
    else:
        fs.put_file(local_path, remote_path)


def get_protocol(url: str) -> str:
    """Return LuxonisFileSystem protocol."""
    return LuxonisFileSystem.get_protocol(url)
