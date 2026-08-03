"""Host-side staging of user-provided inputs into the hidden cache.

The conversion runs inside a Docker container that only has the cache
directory bind-mounted (at ``/app/shared_with_container``). To let users
reference input files by *any* path on their machine, the launcher
copies every file/directory passed on the CLI into the cache (keyed by a
content hash for de-duplication) and rewrites the corresponding CLI
token to the container-side cache path.

Files referenced *inside* a config file are not visible as CLI tokens,
so a config is parsed instead: every local path it names is staged on
its own and the staged copy of the config points at the container-side
locations. Only the files the conversion actually needs are copied,
whatever the config happens to sit next to.
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

# Config fields holding a local path, mapped to the parent key they must
# appear under (``None`` for any). Taken from the config schema rather than
# from the shape of the value, so a string that merely happens to name an
# existing file -- a stage called after a directory, say -- is left alone.
_PATH_FIELDS: dict[str, frozenset[str] | None] = {
    # `modelconverter.utils.config.SingleStageConfig`
    "input_model": None,
    "input_bin": None,
    # `ImageCalibrationConfig.path`, `LinkCalibrationConfig.script`
    "path": frozenset({"calibration"}),
    "script": frozenset({"calibration"}),
    # `RVC4Config.encodings`
    "encodings": frozenset({"rvc4"}),
}

# Never copied along when a directory is staged: repository metadata is not a
# model input and would dominate the copy.
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

    # Config file: stage the files it references, wherever they live.
    if src.suffix.lower() in _CONFIG_EXTS:
        dest = _stage_config(src, inputs_dir)
        return _to_container(dest)

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
    try:
        external_data = [p for p in get_external_data_paths(src) if p.exists()]
    except Exception as exc:
        # Staging must not be the thing that reports a broken model: copy it
        # and let the conversion fail with a message about the model itself.
        logger.debug(f"Could not read external data locations of {src}: {exc}")
        return _stage_file(src, inputs_dir)

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


def _stage_config(src: Path, inputs_dir: Path) -> Path:
    """Stages a YAML config together with the files it references.

    Each local path named by the config is staged on its own and the
    staged copy points at the container-side location, so a config is
    never a reason to copy the directory it happens to live in.
    """
    try:
        data = yaml.safe_load(src.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        # Not a config we can read. Hand it over as-is and let the container
        # report what is wrong with it.
        return _stage_file(src, inputs_dir)

    rewritten, changed = _rewrite_config_paths(data, src.parent, inputs_dir)
    if not changed:
        return _stage_file(src, inputs_dir)
    return _stage_config_text(
        yaml.safe_dump(rewritten, sort_keys=False), src.name, inputs_dir
    )


def _stage_config_text(content: str, name: str, inputs_dir: Path) -> Path:
    """Publishes a rewritten config, keyed by a digest of its own text.

    The rewritten paths carry the digests of everything the config
    references, so a change to any input yields a new entry instead of
    overwriting one a concurrent run may still be reading.
    """
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    dest = inputs_dir / digest / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(dest, content)
    return dest


def _rewrite_config_paths(
    value: Any,
    config_dir: Path,
    inputs_dir: Path,
    key: str | None = None,
    parent: str | None = None,
) -> tuple[Any, bool]:
    """Stages the local paths ``value`` names and returns it with those
    references replaced by their container-side paths."""
    if isinstance(value, dict):
        changed = False
        rewritten = {}
        for item_key, item in value.items():
            rewritten[item_key], item_changed = _rewrite_config_paths(
                value=item,
                config_dir=config_dir,
                inputs_dir=inputs_dir,
                key=item_key,
                parent=key,
            )
            changed |= item_changed
        return rewritten, changed

    if isinstance(value, list):
        changed = False
        rewritten_list = []
        # A list does not introduce a key of its own: an entry of `inputs`
        # is still reached under the key `inputs`.
        for item in value:
            new_item, item_changed = _rewrite_config_paths(
                value=item,
                config_dir=config_dir,
                inputs_dir=inputs_dir,
                key=key,
                parent=parent,
            )
            rewritten_list.append(new_item)
            changed |= item_changed
        return rewritten_list, changed

    if not isinstance(value, str) or not _is_path_field(key, parent):
        return value, False

    staged = _stage_config_reference(value, config_dir, inputs_dir)
    if staged is None:
        return value, False
    return staged, True


def _is_path_field(key: str | None, parent: str | None) -> bool:
    if key is None or key not in _PATH_FIELDS:
        return False
    parents = _PATH_FIELDS[key]
    return parents is None or parent in parents


def _stage_config_reference(
    value: str, config_dir: Path, inputs_dir: Path
) -> str | None:
    """Stages one path referenced by a config, or returns ``None``.

    Relative references are resolved the way the container resolves
    them: against the config file's directory first, then against the
    default root (see
    L{modelconverter.utils.filesystem_utils.get_input_bases}). A
    reference that exists under neither is left untouched so the config
    validation can report it.
    """
    if get_protocol(value) != "file":
        return None

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        bases = [candidate]
    else:
        bases = [config_dir / candidate, Path.cwd() / candidate]

    for base in bases:
        if not base.exists():
            continue
        if base.is_file() and base.suffix.lower() in _CONFIG_EXTS:
            # No config field points at another config, so a YAML reached from
            # inside one is data. Staging it as such also keeps a config that
            # names itself from recursing forever.
            return _to_container(_stage_file(base.resolve(), inputs_dir))
        return _stage_value(str(base), inputs_dir)
    return None


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


# Warn when a single staged directory is larger than this (a calibration
# dataset pointed at a whole photo library, say).
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
    directory under ``$HOME``, say) would otherwise make the copy
    recurse into its own output. ``./output`` is separately mounted into
    the container and accumulates the results of every previous run.
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
                "The conversion runs in a container that only sees the cache, "
                "so every input directory is copied into it. Run "
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

    A staged directory routinely holds files that have nothing to do
    with the model -- root-owned leftovers of an earlier container run,
    for instance. Failing the conversion over one of those would be
    worse than leaving it out; if it really was an input, the container
    reports it as missing.
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
