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

import ast
import atexit
import errno
import hashlib
import inspect
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Collection, Iterator
from pathlib import Path
from typing import Any, NamedTuple

import psutil
import yaml
from loguru import logger

from modelconverter.utils.constants import CONTAINER_SHARED_DIR, get_cache_dir
from modelconverter.utils.filesystem_utils import get_protocol
from modelconverter.utils.general import human_size
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

# Config fields holding a list of raw tool arguments, mapped to the flags
# within them whose *value* is a local path. The list as a whole is not a path
# field: only the token following one of these flags is staged, so the flags
# themselves and every unrelated argument are passed through untouched.
_PATH_ARG_FIELDS: dict[str, frozenset[str]] = {
    # `RVC4Config.validate_quantization_overrides` opens this file while the
    # config is validated, inside the container.
    "snpe_onnx_to_dlc_args": frozenset({"--quantization_overrides"}),
    # OpenVINO's model optimizer reads the transformation rules from this
    # file. Nothing on the Hub sets it, but a local run can.
    "mo_args": frozenset({"--transformations_config"}),
    # `compile_tool` takes its configuration as a file; the exporter writes
    # one itself only when the user has not passed `-c` already.
    "compile_tool_args": frozenset({"-c"}),
}

# Config fields naming a *destination*. These look exactly like input paths but
# must keep pointing at the user's location: rewriting one would send the
# converted model into the cache instead.
_OUTPUT_FIELDS = {"output_remote_url", "intermediate_outputs_remote_url"}

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

    flags: set[str] = set()
    for name, parameter in parameters.items():
        if parameter.kind is parameter.POSITIONAL_ONLY:
            continue
        if not _is_path_parameter(name, parameter.annotation):
            continue
        # Cyclopts accepts an option under both its dashed and its underscored
        # spelling, so a path passed as `--model_path` has to be staged just
        # like one passed as `--model-path`. Recognising only the dashed form
        # would let the other one through as a host path the container cannot
        # resolve.
        flags.add(f"--{name.replace('_', '-')}")
        flags.add(f"--{name}")
    return flags


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
    if token.startswith("--") and "=" in token:
        flag, _, value = token.partition("=")
        if flag in path_flags:
            staged = _stage_value(value, inputs_dir)
            return f"{flag}={staged}" if staged is not None else None
        return None

    if _is_flag(token):
        return None

    # A value explicitly introduced by a path flag.
    if prev in path_flags:
        return _stage_value(token, inputs_dir)

    # A value introduced by a flag we must not stage, or by an unknown/boolean
    # flag: leave it alone.
    if prev is not None and _is_flag(prev):
        return None

    # A config override (`<key> <value>`). The schema knows what the value is,
    # which beats guessing from its shape: a bare directory name is staged when
    # the key calls for a path, and a destination is never staged at all.
    if prev is not None:
        kind = _override_key_kind(prev)
        if kind == "output":
            return None
        if kind == "path":
            return _stage_value(token, inputs_dir)
        if kind == "args":
            return _stage_arg_list_token(token, prev, inputs_dir)

    # A bare positional token (target, unknown override value, ...). Only stage
    # it when it clearly looks like an existing local path.
    if _is_path_like(token):
        return _stage_value(token, inputs_dir)
    return None


def _is_flag(token: str) -> bool:
    """Whether ``token`` is an option rather than a value.

    Single-dash aliases count: the value after ``-c`` is no more an input
    path than one after ``--config``. A negative number is a value,
    whether it leads with a digit (``-5``) or a decimal point (``-.5``).
    """
    if len(token) <= 1 or token[0] != "-" or token[1].isdigit():
        return False
    return not (token[1] == "." and len(token) > 2 and token[2].isdigit())


def _override_key_kind(key: str) -> str | None:
    """Classifies a config-override key by what its value names.

    Overrides are dotted paths into the config (``calibration.path``), so
    the same schema tables that drive config-file staging apply here.
    """
    head, _, leaf = key.rpartition(".")
    parent = head.rsplit(".", 1)[-1] or None
    # The leaf, not the whole key: `stages.<name>.output_remote_url` names a
    # destination just as much as the bare field does.
    if leaf in _OUTPUT_FIELDS:
        return "output"
    if leaf in _PATH_ARG_FIELDS:
        return "args"
    if _is_path_field(leaf, parent):
        return "path"
    return None


def _stage_arg_list_token(
    token: str, key: str, inputs_dir: Path
) -> str | None:
    """Stages the paths inside a serialized argument-list override.

    The whole list arrives as a single CLI token, so the path it holds is
    invisible to token staging -- while the same list written in a config
    file is rewritten by L{_rewrite_arg_list}. ``LuxonisConfig`` reads
    these values with ``ast.literal_eval`` and falls back to the raw
    string, so reading them the same way here (and writing the result
    back with ``repr``) keeps the two forms behaving alike.

    Relative paths resolve against the working directory: the user typed
    this one in a shell, not inside a config file.
    """
    try:
        parsed = ast.literal_eval(token)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(parsed, list):
        return None

    rewritten, changed = _rewrite_arg_list(
        parsed,
        _PATH_ARG_FIELDS[key.rpartition(".")[2]],
        Path.cwd(),
        inputs_dir,
    )
    return repr(rewritten) if changed else None


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
    p = _as_path(value)
    if p is None or not _exists(p):
        return False
    if p.is_absolute() or "/" in value or "\\" in value:
        return True
    if value.startswith(("~", ".")):
        return True
    return p.is_file() and p.suffix.lower() in _KNOWN_EXTS


def _as_path(value: str) -> Path | None:
    """Interprets ``value`` as a local path, or returns ``None``.

    A CLI override or a config field may legitimately hold something that
    is not a path at all -- an inline encodings JSON, a calibration
    script -- and C{Path} rejects some of those outright.
    """
    try:
        return Path(value).expanduser()
    except (OSError, ValueError, RuntimeError):
        return None


def _absolute(path: Path) -> Path:
    """Absolutizes ``path`` without resolving symlinks.

    Deliberately not C{Path.resolve}: the staged copy keeps the name the
    user asked for. Resolving would name it after the symlink's target,
    and a content-addressed blob (DVC, git-annex, Nix) carries neither
    the model's name nor the suffix the container dispatches on.
    """
    return Path(os.path.abspath(path))  # noqa: PTH100


def _exists(path: Path) -> bool:
    """C{Path.exists} for values that may not name a path.

    An inline JSON blob has no C{/} but is far longer than a filename may
    be, so C{exists} raises C{OSError} (C{ENAMETOOLONG}) instead of
    answering C{False}.
    """
    try:
        return path.exists()
    except (OSError, ValueError):
        return False


def _stage_value(value: str, inputs_dir: Path) -> str | None:
    """Copies the file/dir at ``value`` into the cache and returns the
    container-side path, or ``None`` if it cannot/should not be
    staged."""
    if get_protocol(value) != "file":
        return None
    src = _as_path(value)
    if src is None or not _exists(src):
        return None
    src = _absolute(src)

    if src.is_dir():
        dest = _stage_dir(src, inputs_dir)
        return _to_container(dest)

    if src.suffix.lower() in _CONFIG_EXTS:
        dest = _stage_config(src, inputs_dir)
        return _to_container(dest)

    # An OpenVINO IR is a `.xml`/`.bin` pair; whichever half was named, the
    # container needs both.
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
    digest = _hash_file_memoized(src)
    dest = inputs_dir / digest / src.name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _log_copy(src, src.stat().st_size)
        _atomic_copy_file(src, dest)
    _mark_in_use(dest.parent)
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
        # The container can then only report the missing tensor files, so say
        # here what the actual cause was.
        logger.warning(
            f"Could not read external data locations of {src}: {exc}. "
            "Staging the model file on its own."
        )
        return _stage_file(src, inputs_dir)

    digest = _hash_files([src, *external_data])
    hash_dir = inputs_dir / digest
    destinations = {src: Path(src.name)}
    # `get_external_data_paths` resolves each location against -- and confines
    # it to -- the resolved model directory, so that is what the recorded
    # locations are relative to.
    model_dir = src.parent.resolve()
    for data_path in external_data:
        destinations[data_path] = data_path.relative_to(model_dir)

    _stage_file_bundle(destinations, hash_dir)
    return hash_dir / src.name


def _atomic_copy_file(src: Path, dest: Path) -> None:
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
        _mark_in_use(dest_dir)
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
    _mark_in_use(dest_dir)


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
    _mark_in_use(dest.parent)
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
        if key is not None and key in _PATH_ARG_FIELDS:
            return _rewrite_arg_list(
                value, _PATH_ARG_FIELDS[key], config_dir, inputs_dir
            )
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


def _rewrite_arg_list(
    args: list[Any],
    path_flags: frozenset[str],
    config_dir: Path,
    inputs_dir: Path,
) -> tuple[list[Any], bool]:
    """Stages the values ``path_flags`` introduce in a raw argument list.

    The list is passed to the conversion tool verbatim, so everything
    else in it -- the flags themselves, numeric options, tool-specific
    switches -- has to survive untouched.
    """
    rewritten: list[Any] = []
    changed = False
    for index, item in enumerate(args):
        previous = args[index - 1] if index else None
        staged = None
        if isinstance(item, str):
            if previous in path_flags:
                staged = _stage_config_reference(item, config_dir, inputs_dir)
            elif "=" in item and item.partition("=")[0] in path_flags:
                # The `--flag=value` spelling carries the path in the same
                # token as the flag.
                flag, _, value = item.partition("=")
                staged_value = _stage_config_reference(
                    value, config_dir, inputs_dir
                )
                if staged_value is not None:
                    staged = f"{flag}={staged_value}"
        if staged is None:
            rewritten.append(item)
        else:
            rewritten.append(staged)
            changed = True
    return rewritten, changed


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

    # `calibration.script` may hold the script itself rather than a path to
    # one, so the value is not necessarily something `Path` can represent.
    candidate = _as_path(value)
    if candidate is None:
        return None
    if candidate.is_absolute():
        bases = [candidate]
    else:
        bases = [config_dir / candidate, Path.cwd() / candidate]

    for base in bases:
        if not _exists(base):
            continue
        if base.is_file() and base.suffix.lower() in _CONFIG_EXTS:
            # No config field points at another config, so a YAML reached from
            # inside one is data. Staging it as such also keeps a config that
            # names itself from recursing forever.
            return _to_container(_stage_file(_absolute(base), inputs_dir))
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
            # `replace` only makes the rename atomic; without this the entry
            # can survive a power loss under its digest with truncated
            # contents, and nothing ever re-writes it.
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


# Warn when a single staged directory is larger than this (a calibration
# dataset pointed at a whole photo library, say).
_LARGE_DIR_BYTES = 1024**3  # 1 GiB

# Written next to a staged directory to record the source it was copied from,
# so the previous copy can be dropped once that source changes.
_SOURCE_MARKER = ".source"

# Written inside a staged entry (and at the cache root, for the container's
# own downloads) for as long as this process may be using it, suffixed with
# the pid so a concurrent run can tell live users from the leftovers of a
# killed one.
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
    ends up seeing is also what gets fingerprinted. A link that points
    back at a directory already on the way down -- ``calib/all -> ..``,
    say -- is skipped, which terminates any cycle without dropping two
    sibling links to the same directory: those are distinct content the
    container expects to find under both names. Anything that is not a
    regular file (dangling symlink, socket, fifo) is left out.

    So is a file this process cannot read: a staged directory routinely
    holds files that have nothing to do with the model -- root-owned
    leftovers of an earlier container run, for instance -- and failing
    the conversion over one of those would be worse than leaving it out.
    Leaving it out *here* rather than at copy time keeps the digest a
    description of what the cache entry actually holds, so making the
    file readable later yields a new entry instead of silently reusing
    the one that lacks it.
    """
    excluded = _excluded_dirs()
    # The chain of directory identities leading to each pending directory. It
    # starts with `src` and everything above it, so `calib/latest -> .` and
    # `calib/all -> ..` are both skipped rather than pulling the surrounding
    # tree -- potentially the whole home directory -- into the cache.
    chains: dict[Path, frozenset[tuple[int, int]]] = {
        src: frozenset(
            key
            for directory in (src, *src.parents)
            if (key := _dir_key(directory)) is not None
        )
    }

    for root, dir_names, file_names in os.walk(src, followlinks=True):
        root_path = Path(root)
        ancestors = chains.pop(root_path, frozenset())
        kept = []
        for name in dir_names:
            if name in _IGNORED_DIR_NAMES:
                continue
            directory = root_path / name
            if directory.resolve() in excluded:
                continue
            key = _dir_key(directory)
            if key is None or key in ancestors:
                continue
            chains[directory] = ancestors | {key}
            kept.append(name)
        dir_names[:] = kept
        for name in file_names:
            path = root_path / name
            try:
                st = path.stat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if not os.access(path, os.R_OK):
                logger.warning(f"Skipping unreadable file {path}")
                continue
            yield _InputFile(path.relative_to(src).as_posix(), path, st)


def _dir_key(directory: Path) -> tuple[int, int] | None:
    """The identity a symlink cycle would repeat, or ``None`` if the
    directory cannot be stat'ed."""
    try:
        st = directory.stat()
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _stage_dir(src: Path, inputs_dir: Path) -> Path:
    files = sorted(_iter_input_files(src), key=lambda f: f.relative)
    digest = _hash_dir(src, files)
    digest_dir = inputs_dir / digest
    dest = digest_dir / src.name
    if not dest.exists():
        size = sum(f.stat.st_size for f in files)
        if size > _LARGE_DIR_BYTES:
            logger.warning(
                f"Caching a large directory {src} ({human_size(size)}). "
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

    Unreadable files were already left out by L{_iter_input_files}, so a
    failure here means the source changed since it was enumerated. It is
    raised rather than logged: the digest promises the whole enumeration,
    and publishing less than that under it would hide the missing file
    from every later run.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for file in files:
        target = dest / file.relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file.path, target)


def _record_source(src: Path, digest_dir: Path) -> None:
    marker = digest_dir / _SOURCE_MARKER
    if not marker.exists():
        _atomic_write_text(marker, str(src))


def _mark_in_use(digest_dir: Path) -> None:
    """Claims ``digest_dir`` for the lifetime of this process.

    The marker records the process creation time alongside the pid: the
    pid alone would keep the claim alive once an unrelated process
    recycles it after this one is killed.
    """
    marker = digest_dir / f"{_IN_USE_PREFIX}{os.getpid()}"
    try:
        _atomic_write_text(marker, str(psutil.Process().create_time()))
    except (OSError, psutil.Error):
        return
    atexit.register(marker.unlink, missing_ok=True)


def _is_in_use(entry: Path) -> bool:
    """Whether a live process has claimed ``entry``.

    The container reads its staged inputs throughout the conversion, so
    an entry another run is still using must survive even once its
    source has changed. Claims left behind by a killed process name a
    pid that no longer exists -- or one that has since been recycled by
    an unrelated process, which the recorded creation time tells apart --
    and are cleaned up here.
    """
    in_use = False
    for marker in entry.glob(f"{_IN_USE_PREFIX}*"):
        try:
            pid = int(marker.name.removeprefix(_IN_USE_PREFIX))
        except ValueError:
            continue
        try:
            create_time = psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            marker.unlink(missing_ok=True)
            continue
        except psutil.Error:
            # The pid exists but is not ours to inspect.
            in_use = True
            continue
        try:
            recorded = float(marker.read_text())
        except (OSError, ValueError):
            # No recorded time to compare against: assume the claim holds.
            in_use = True
            continue
        # A recycled pid belongs to a process started well after the claim
        # was written; the tolerance only absorbs clock rounding.
        if abs(recorded - create_time) < 1.0:
            in_use = True
        else:
            marker.unlink(missing_ok=True)
    return in_use


def in_use_staged_inputs() -> list[Path]:
    """Returns the staged input entries a live process has claimed.

    The container reads its staged inputs throughout the conversion, so
    anything that empties the cache wholesale (``cache clean``) has to be
    able to see that a run in another terminal is still using it.
    """
    try:
        entries = sorted((get_cache_dir() / "inputs").iterdir())
    except OSError:
        return []
    return [entry for entry in entries if entry.is_dir() and _is_in_use(entry)]


def claim_cache() -> None:
    """Claims the whole cache for the lifetime of this process.

    The container downloads remote models and calibration data straight
    into the cache mount as it runs, so a conversion can be using the
    cache even when none of its inputs were staged host-side. The
    launcher places this claim before starting the container.
    """
    cache_dir = get_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    _mark_in_use(cache_dir)


def cache_is_in_use() -> bool:
    """Whether a live process is still using any part of the cache."""
    return _is_in_use(get_cache_dir()) or bool(in_use_staged_inputs())


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
    """Maps a host cache path to its container-side location.

    The container runs Linux whatever the host is, so the rewritten
    token must keep forward slashes even when the host's native paths
    do not.
    """
    relative = dest.relative_to(get_cache_dir())
    return str(CONTAINER_SHARED_DIR / relative.as_posix())


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _hash_file_memoized(path: Path) -> str:
    """Content digest of ``path``, memoized on its stat fingerprint.

    Staging would otherwise re-read every staged file in full on each
    run just to learn that the cache already holds it -- minutes of
    start-up I/O for a multi-gigabyte ONNX model. The memo key carries
    the same metadata as L{_hash_dir} (and shares its caveat about
    filesystems without a change time), while the digest the cache is
    keyed on stays a description of the bytes.
    """
    try:
        st = path.stat()
    except OSError:
        return _hash_file(path)
    key = hashlib.sha256(
        f"{path}\0{st.st_size}\0{st.st_mtime_ns}\0{st.st_ctime_ns}\0"
        f"{st.st_dev}\0{st.st_ino}".encode()
    ).hexdigest()[:16]
    memo = get_cache_dir() / "digests" / key
    try:
        digest = memo.read_text()
    except OSError:
        digest = ""
    if _is_digest(digest):
        return digest
    digest = _hash_file(path)
    try:
        memo.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(memo, digest)
    except OSError:
        pass
    return digest


def _is_digest(value: str) -> bool:
    if len(value) != 16:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _hash_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.name.encode())
        h.update(_hash_file_memoized(path).encode())
    return h.hexdigest()[:16]


def _hash_dir(path: Path, files: list[_InputFile] | None = None) -> str:
    """Fast fingerprint of a directory from the metadata of the files
    that are staged, rather than from their contents.

    Reading a calibration set in full on every run would cost more than
    the conversion, but the size and modification time alone are too weak
    to key a cache on: restoring a backup or an ``rsync -a`` preserves
    both, and the container would then quantize against the previous
    images without any sign that it had. The inode and the change time
    are stat'ed anyway and neither survives a file being replaced, so
    they are folded in as well -- with the device, since an inode number
    is only unique within one filesystem.

    This is still a description of the metadata rather than of the bytes:
    a filesystem that does not maintain a change time (FAT, some network
    mounts) can hide an in-place edit that keeps the size and the
    modification time.
    """
    if files is None:
        files = sorted(_iter_input_files(path), key=lambda f: f.relative)
    h = hashlib.sha256()
    h.update(path.name.encode())
    for file in files:
        h.update(
            f"{file.relative}\0{file.stat.st_size}\0"
            f"{file.stat.st_mtime_ns}\0{file.stat.st_ctime_ns}\0"
            f"{file.stat.st_dev}\0{file.stat.st_ino}".encode()
        )
    return h.hexdigest()[:16]


def _log_copy(src: Path, size: int) -> None:
    logger.info(f"Caching input {src} ({human_size(size)})")
