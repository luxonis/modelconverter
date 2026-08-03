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

import atexit
import errno
import hashlib
import inspect
import os
import shutil
import stat as stat_module
import tempfile
from collections.abc import Callable, Collection, Iterator
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from loguru import logger

from modelconverter.utils.constants import CONTAINER_SHARED_DIR, get_cache_dir
from modelconverter.utils.filesystem_utils import get_protocol
from modelconverter.utils.onnx_compatibility import get_external_data_paths

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

# Config keys naming an output *destination* rather than an input. Staging one
# would redirect the results into the throwaway cache directory.
_DESTINATION_KEYS = {
    "output_remote_url",
    "intermediate_outputs_remote_url",
    "output_dir",
}

# Never copied along when a config's whole parent directory is staged:
# repository metadata is not a model input and would dominate the copy.
_IGNORED_DIR_NAMES = {".git"}


def path_flags_for(command: Callable[..., Any] | None) -> set[str]:
    """Returns the CLI flags of ``command`` whose value is a local path.

    Derived from the command signature the launcher has already parsed, so
    staging follows the CLI rather than duplicating its knowledge: a path
    option that is added or renamed is picked up without touching this module.
    """
    try:
        parameters = inspect.signature(command).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return set()

    return {
        f"--{name.replace('_', '-')}"
        for name, parameter in parameters.items()
        if parameter.kind is not parameter.POSITIONAL_ONLY
        and _is_path_parameter(name, parameter.annotation)
    }


def _is_path_parameter(name: str, annotation: Any) -> bool:
    return (
        name in {"path", "config"}
        or name.endswith("_path")
        or annotation is Path
    )


def stage_inputs(tokens: list[str], path_flags: Collection[str]) -> list[str]:
    """Copies path arguments among ``tokens`` into the cache and returns
    a new token list with those paths rewritten to their container-side
    locations.

    ``path_flags`` are the flags whose value is a path, as returned by
    L{path_flags_for}. Non-path tokens, remote URLs and non-existent
    paths are left untouched.
    """
    inputs_dir = get_cache_dir() / "inputs"
    new_tokens: list[str] = []
    prev: str | None = None

    for token in tokens:
        staged = _maybe_stage_token(token, prev, inputs_dir, path_flags)
        new_tokens.append(staged if staged is not None else token)
        prev = token

    return new_tokens


def _maybe_stage_token(
    token: str,
    prev: str | None,
    inputs_dir: Path,
    path_flags: Collection[str],
) -> str | None:
    """Returns the rewritten token if it should be staged, else
    ``None``."""
    # Handle the ``--flag=value`` form.
    if token.startswith("--") and "=" in token:
        flag, _, value = token.partition("=")
        if flag in path_flags:
            staged = _stage_value(value, inputs_dir)
            return f"{flag}={staged}" if staged is not None else None
        return None

    if token.startswith("--"):
        return None

    # A value explicitly introduced by a path flag.
    if prev in path_flags:
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
    """Whether a bare positional token should be treated as a local
    path.

    Bare names are ambiguous: a config-override value such as
    ``resnet18`` may well collide with a directory of the same name in
    the current directory, and rewriting it would silently corrupt the
    override. A token therefore only counts as a path when its *shape*
    says so, or when it is a file with a known model/config extension.
    Anything else has to be written with a ``./`` prefix to be staged.
    """
    if value in _TARGET_NAMES:
        return False
    if get_protocol(value) != "file":
        return False
    p = Path(value).expanduser()
    if not p.exists():
        return False
    if p.is_absolute() or "/" in value or "\\" in value:
        return True
    if value.startswith(("~", ".")):
        return True
    return p.is_file() and p.suffix.lower() in _KNOWN_EXTS


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
    # references resolve alongside it, then stage and rewrite absolute local
    # references which may live anywhere on the host.
    if src.suffix.lower() in _CONFIG_EXTS:
        parent_dest = _stage_dir(src.parent, inputs_dir)
        config_dest = parent_dest / src.name
        _rewrite_absolute_config_paths(src, config_dest, inputs_dir)
        return _to_container(config_dest)

    # OpenVINO IR: stage the `.xml`/`.bin` pair together.
    if src.suffix.lower() in {".xml", ".bin"}:
        dest = _stage_ir_pair(src, inputs_dir)
        return _to_container(dest)

    # ONNX models may store their tensors in companion files. Preserve the
    # relative external-data locations expected by the model.
    if src.suffix.lower() == ".onnx":
        dest = _stage_onnx(src, inputs_dir)
        return _to_container(dest)

    dest = _stage_file(src, inputs_dir)
    return _to_container(dest)


def _stage_file(src: Path, inputs_dir: Path) -> Path:
    digest = _hash_file(src)
    dest = inputs_dir / digest / src.name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _log_copy(src, src.stat().st_size)
        _atomic_copy_file(src, dest)
    return dest


def _stage_ir_pair(src: Path, inputs_dir: Path) -> Path:
    xml = src.with_suffix(".xml")
    bin_ = src.with_suffix(".bin")
    members = [p for p in (xml, bin_) if p.exists()]
    digest = _hash_files(members)
    hash_dir = inputs_dir / digest
    destinations = {member: Path(member.name) for member in members}
    _stage_file_bundle(destinations, hash_dir)
    return hash_dir / src.name


def _stage_onnx(src: Path, inputs_dir: Path) -> Path:
    # A model saved with `all_tensors_to_one_file=False` has one companion
    # file per tensor, so every location has to be staged, not just the first.
    external_data = [p for p in get_external_data_paths(src) if p.exists()]

    digest = _hash_files([src, *external_data])
    hash_dir = inputs_dir / digest
    destinations = {src: Path(src.name)}
    for data_path in external_data:
        destinations[data_path] = data_path.relative_to(src.parent)

    _stage_file_bundle(destinations, hash_dir)
    return hash_dir / src.name


def _atomic_copy_file(src: Path, dest: Path) -> None:
    """Copy one file and atomically publish it at ``dest``."""
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.tmp-", dir=dest.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(src, tmp)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _stage_file_bundle(destinations: dict[Path, Path], dest_dir: Path) -> None:
    """Atomically stage a set of files as one digest-keyed directory."""
    if dest_dir.exists():
        return

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{dest_dir.name}.tmp-", dir=dest_dir.parent)
    )
    try:
        for src, relative_dest in destinations.items():
            tmp_dest = tmp_dir / relative_dest
            tmp_dest.parent.mkdir(parents=True, exist_ok=True)
            _log_copy(src, src.stat().st_size)
            shutil.copy2(src, tmp_dest)
        _publish_directory(tmp_dir, dest_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _publish_directory(src: Path, dest: Path) -> None:
    """Rename ``src`` to ``dest``, accepting a concurrent winner."""
    try:
        src.rename(dest)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise


def _rewrite_absolute_config_paths(
    src: Path, dest: Path, inputs_dir: Path
) -> None:
    """Stage absolute local paths in a YAML config and rewrite its
    copy."""
    data = yaml.safe_load(src.read_text())
    rewritten, changed = _rewrite_absolute_paths(data, inputs_dir)
    if changed:
        _atomic_write_text(dest, yaml.safe_dump(rewritten, sort_keys=False))
    else:
        # The cached directory may have been used for a previously rewritten
        # config. Restore the source verbatim when no references are rewritten.
        _atomic_copy_file(src, dest)


def _atomic_write_text(dest: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.tmp-", dir=dest.parent, text=True
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _rewrite_absolute_paths(value: Any, inputs_dir: Path) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        rewritten = {}
        for key, item in value.items():
            # An output destination is not an input: staging it would send the
            # results into the cache instead of where the user asked for them.
            if key in _DESTINATION_KEYS:
                rewritten[key] = item
                continue
            new_item, item_changed = _rewrite_absolute_paths(item, inputs_dir)
            rewritten[key] = new_item
            changed |= item_changed
        return rewritten, changed

    if isinstance(value, list):
        changed = False
        rewritten_list = []
        for item in value:
            new_item, item_changed = _rewrite_absolute_paths(item, inputs_dir)
            rewritten_list.append(new_item)
            changed |= item_changed
        return rewritten_list, changed

    if isinstance(value, str):
        path = Path(value).expanduser()
        if path.is_absolute() and path.exists():
            staged = _stage_value(value, inputs_dir)
            if staged is not None:
                return staged, True

    return value, False


# Warn when a single staged directory is larger than this (e.g. a config file
# that happens to live in a big folder, whose whole parent gets copied).
_LARGE_DIR_BYTES = 1024**3  # 1 GiB

# Written next to a staged directory to record the source it was copied from,
# so the previous copy can be dropped once that source changes.
_SOURCE_MARKER = ".source"

# Written next to a staged directory for as long as this process may be using
# it, suffixed with the pid so a concurrent run can tell live users from the
# leftovers of a killed one.
_IN_USE_PREFIX = ".inuse-"


class _InputFile(NamedTuple):
    relative: str
    path: Path
    stat: os.stat_result


def _excluded_dirs() -> set[Path]:
    """Absolute directories that must never be copied into the cache.

    The cache is the copy *destination*, so a source containing it (a
    config kept in ``$HOME``, say) would otherwise make the copy recurse
    into its own output. ``./output`` is separately mounted into the
    container and accumulates the results of every previous run.
    """
    return {
        get_cache_dir().resolve(),
        (Path.cwd() / "output").resolve(),
    }


def _iter_input_files(src: Path) -> Iterator[_InputFile]:
    """Yields every file that staging ``src`` copies, with its stat.

    Symlinked sub-directories are followed so that what the container
    ends up seeing is also what gets fingerprinted; a directory already
    visited is skipped so a symlink cycle terminates. Anything that is
    not a regular file (dangling symlink, socket, fifo) is left out.
    """
    excluded = _excluded_dirs()
    visited: set[tuple[int, int]] = set()

    for root, dir_names, file_names in os.walk(src, followlinks=True):
        root_path = Path(root)
        dir_names[:] = [
            name
            for name in dir_names
            if name not in _IGNORED_DIR_NAMES
            and (root_path / name).resolve() not in excluded
            and _first_visit(root_path / name, visited)
        ]
        for name in file_names:
            path = root_path / name
            try:
                file_stat = path.stat()
            except OSError:
                continue
            if stat_module.S_ISREG(file_stat.st_mode):
                yield _InputFile(
                    path.relative_to(src).as_posix(), path, file_stat
                )


def _first_visit(directory: Path, visited: set[tuple[int, int]]) -> bool:
    try:
        directory_stat = directory.stat()
    except OSError:
        return False
    key = (directory_stat.st_dev, directory_stat.st_ino)
    if key in visited:
        return False
    visited.add(key)
    return True


def _stage_dir(src: Path, inputs_dir: Path) -> Path:
    files = sorted(_iter_input_files(src), key=lambda f: f.relative)
    digest = _hash_dir(src, files)
    digest_dir = inputs_dir / digest
    dest = digest_dir / src.name
    if not dest.exists():
        size = sum(f.stat.st_size for f in files)
        if size > _LARGE_DIR_BYTES:
            logger.warning(
                f"Caching a large directory {src} ({_human_size(size)}). "
                "Local files referenced by a config are resolved relative to "
                "the config file, so its whole directory is copied. Consider "
                "keeping configs in a dedicated folder. Run "
                "`modelconverter cache clean` to reclaim space."
            )
        _log_copy(src, size)
        digest_dir.mkdir(parents=True, exist_ok=True)
        tmp_root = Path(
            tempfile.mkdtemp(prefix=f".{src.name}.tmp-", dir=digest_dir)
        )
        tmp_dest = tmp_root / src.name
        try:
            _copy_input_files(files, tmp_dest)
            _publish_directory(tmp_dest, dest)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
    _record_source(src, digest_dir)
    _mark_in_use(digest_dir)
    _prune_superseded_stagings(src, digest_dir, inputs_dir)
    return dest


def _copy_input_files(files: list[_InputFile], dest: Path) -> None:
    """Copies the enumerated files under ``dest``.

    A config's whole parent directory is staged, so it routinely holds
    files that have nothing to do with the model -- root-owned leftovers
    of an earlier container run, for instance. Failing the conversion
    over one of those would be worse than leaving it out; if it really
    was an input, the container reports it as missing.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for file in files:
        target = dest / file.relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(file.path, target)
        except OSError as exc:
            logger.warning(f"Skipping unreadable file {file.path}: {exc}")


def _record_source(src: Path, digest_dir: Path) -> None:
    marker = digest_dir / _SOURCE_MARKER
    if not marker.exists():
        _atomic_write_text(marker, str(src))


def _mark_in_use(digest_dir: Path) -> None:
    """Claims ``digest_dir`` for the lifetime of this process."""
    marker = digest_dir / f"{_IN_USE_PREFIX}{os.getpid()}"
    try:
        marker.touch()
    except OSError:
        return
    atexit.register(marker.unlink, missing_ok=True)


def _is_in_use(entry: Path) -> bool:
    """Whether a live process has claimed ``entry``.

    The container reads its staged inputs throughout the conversion, so
    an entry another run is still using must survive even once its
    source has changed. Claims left behind by a killed process name a
    pid that no longer exists and are cleaned up here.
    """
    in_use = False
    for marker in entry.glob(f"{_IN_USE_PREFIX}*"):
        try:
            pid = int(marker.name.removeprefix(_IN_USE_PREFIX))
        except ValueError:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            marker.unlink(missing_ok=True)
        except OSError:
            # The pid exists but is not ours to signal.
            in_use = True
        else:
            in_use = True
    return in_use


def _prune_superseded_stagings(
    src: Path, keep: Path, inputs_dir: Path
) -> None:
    """Removes staged copies of ``src`` made for older content.

    A directory digest covers file modification times, so editing
    anything under ``src`` yields a new entry; without this the cache
    would keep every copy it ever made of that directory.
    """
    source = str(src)
    for entry in inputs_dir.iterdir():
        if entry == keep or not entry.is_dir():
            continue
        try:
            if (entry / _SOURCE_MARKER).read_text() != source:
                continue
        except OSError:
            continue
        if _is_in_use(entry):
            continue
        logger.debug(f"Removing superseded staged copy of {src} at {entry}")
        shutil.rmtree(entry, ignore_errors=True)


def _to_container(dest: Path) -> str:
    """Maps a host cache path to its container-side location."""
    relative = dest.relative_to(get_cache_dir())
    return str(CONTAINER_SHARED_DIR / relative)


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


def _hash_dir(path: Path, files: list[_InputFile] | None = None) -> str:
    """Fast content-fingerprint of a directory based on the relative
    paths, sizes and modification times of the files that are staged."""
    if files is None:
        files = sorted(_iter_input_files(path), key=lambda f: f.relative)
    h = hashlib.sha256()
    h.update(path.name.encode())
    for file in files:
        h.update(
            f"{file.relative}\0{file.stat.st_size}\0"
            f"{file.stat.st_mtime_ns}".encode()
        )
    return h.hexdigest()[:16]


def _log_copy(src: Path, size: int) -> None:
    logger.info(f"Caching input {src} ({_human_size(size)})")


def _human_size(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PiB"
