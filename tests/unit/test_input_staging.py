import ast
import errno
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import yaml

from modelconverter.utils import input_staging
from modelconverter.utils.constants import CONTAINER_SHARED_DIR
from tests.helpers.onnx_factory import (
    single_io_onnx,
    weights_only_external_onnx,
)


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Rebases the staging cache into the test's temp directory."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache)
    return cache


def _host_staged_path(staged: str, cache_dir: Path) -> Path:
    relative = Path(staged).relative_to(CONTAINER_SHARED_DIR)
    return cache_dir / relative


def _external_onnx(path: Path) -> Path:
    (external_data,) = weights_only_external_onnx(path)
    return external_data


def test_stages_onnx_external_data(tmp_path: Path, cache_dir: Path) -> None:
    model_path = tmp_path / "models" / "model.onnx"
    model_path.parent.mkdir()
    external_data = _external_onnx(model_path)

    staged_tokens = input_staging.stage_inputs(
        ["--model-path", str(model_path)], {"--model-path"}
    )

    staged_model = _host_staged_path(staged_tokens[1], cache_dir)
    assert staged_model.read_bytes() == model_path.read_bytes()
    assert staged_model.with_name(external_data.name).read_bytes() == (
        external_data.read_bytes()
    )


def test_stages_every_external_data_file(
    tmp_path: Path, cache_dir: Path
) -> None:
    model_path = tmp_path / "models" / "model.onnx"
    model_path.parent.mkdir()
    external_data = weights_only_external_onnx(model_path, single_file=False)
    assert len(external_data) > 1

    staged_tokens = input_staging.stage_inputs(
        ["--model-path", str(model_path)], {"--model-path"}
    )

    staged_model = _host_staged_path(staged_tokens[1], cache_dir)
    for data_path in external_data:
        assert staged_model.with_name(data_path.name).read_bytes() == (
            data_path.read_bytes()
        )


def _staged_content(staged: str, cache_dir: Path) -> bytes:
    return _host_staged_path(staged, cache_dir).read_bytes()


def test_stages_and_rewrites_absolute_config_paths(
    tmp_path: Path, cache_dir: Path
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    absolute_model = tmp_path / "models" / "absolute.onnx"
    absolute_model.parent.mkdir()
    absolute_model.write_bytes(b"model")
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "image.jpg").write_bytes(b"image")
    script = tmp_path / "scripts" / "calibration.py"
    script.parent.mkdir()
    script.write_text("print('calibrate')")
    encodings = tmp_path / "encodings.json"
    encodings.write_text("{}")

    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input_model": str(absolute_model),
                "calibration": {"path": str(calibration)},
                "inputs": [
                    {
                        "name": "images",
                        "calibration": {
                            "stage": "first",
                            "script": str(script),
                        },
                    }
                ],
                "rvc4": {"encodings": str(encodings)},
                "remote_model": "s3://bucket/model.onnx",
            }
        )
    )

    staged_tokens = input_staging.stage_inputs(
        ["--config", str(config_path)], {"--config"}
    )

    staged_config = _host_staged_path(staged_tokens[1], cache_dir)
    rewritten = yaml.safe_load(staged_config.read_text())
    assert _staged_content(rewritten["input_model"], cache_dir) == b"model"
    assert (
        _staged_content(
            rewritten["inputs"][0]["calibration"]["script"], cache_dir
        )
        == b"print('calibrate')"
    )
    assert _staged_content(rewritten["rvc4"]["encodings"], cache_dir) == b"{}"
    staged_calibration = _host_staged_path(
        rewritten["calibration"]["path"], cache_dir
    )
    assert (staged_calibration / "image.jpg").read_bytes() == b"image"
    assert rewritten["remote_model"] == "s3://bucket/model.onnx"


def test_stages_and_rewrites_relative_config_paths(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    (config_dir / "calibration_data").mkdir(parents=True)
    (config_dir / "model.onnx").write_bytes(b"model")
    (config_dir / "calibration_data" / "image.jpg").write_bytes(b"image")
    (config_dir / "script.py").write_text("print('calibrate')")

    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "stages": {
                    "first": {
                        "input_model": "model.onnx",
                        "calibration": {"path": "calibration_data"},
                    },
                    "second": {
                        "input_model": "model.onnx",
                        "inputs": [
                            {
                                "name": "images",
                                "calibration": {
                                    "stage": "first",
                                    "script": "script.py",
                                },
                            }
                        ],
                    },
                }
            }
        )
    )
    # The config directory is not the working directory: relative references
    # have to resolve against the config file itself.
    monkeypatch.chdir(tmp_path)

    staged_tokens = input_staging.stage_inputs(
        ["--path", str(config_path)], {"--path"}
    )

    rewritten = yaml.safe_load(
        _host_staged_path(staged_tokens[1], cache_dir).read_text()
    )
    stages = rewritten["stages"]
    assert (
        _staged_content(stages["first"]["input_model"], cache_dir) == b"model"
    )
    calibration = _host_staged_path(
        stages["first"]["calibration"]["path"], cache_dir
    )
    assert (calibration / "image.jpg").read_bytes() == b"image"
    assert (
        _staged_content(
            stages["second"]["inputs"][0]["calibration"]["script"], cache_dir
        )
        == b"print('calibrate')"
    )


def test_only_config_references_are_staged(
    tmp_path: Path, cache_dir: Path
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "model.onnx").write_bytes(b"model")
    # A config routinely sits in a directory holding much more than the
    # conversion needs; none of it may end up in the cache.
    (config_dir / "unrelated.bin").write_bytes(b"unrelated")
    (config_dir / "notes").mkdir()
    (config_dir / "notes" / "todo.txt").write_text("todo")

    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump({"input_model": "model.onnx"}))

    input_staging.stage_inputs(["--path", str(config_path)], {"--path"})

    staged_names = {
        path.name
        for path in cache_dir.rglob("*")
        if path.is_file()
        # The cache holds bookkeeping of its own -- in-use claims and
        # digest memos -- which is not staged content.
        and not path.name.startswith(".")
        and path.parent.name != "digests"
    }
    assert staged_names == {"config.yaml", "model.onnx"}


def test_stages_the_openvino_bin_alongside_the_xml(
    tmp_path: Path, cache_dir: Path
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    (ir_dir / "model.xml").write_bytes(b"xml")
    (ir_dir / "model.bin").write_bytes(b"weights")

    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input_model": str(ir_dir / "model.xml"),
                "input_bin": str(ir_dir / "model.bin"),
            }
        )
    )

    staged_tokens = input_staging.stage_inputs(
        ["--path", str(config_path)], {"--path"}
    )

    rewritten = yaml.safe_load(
        _host_staged_path(staged_tokens[1], cache_dir).read_text()
    )
    staged_xml = _host_staged_path(rewritten["input_model"], cache_dir)
    assert staged_xml.read_bytes() == b"xml"
    assert staged_xml.with_suffix(".bin").read_bytes() == b"weights"
    assert _staged_content(rewritten["input_bin"], cache_dir) == b"weights"


def test_missing_config_references_are_left_untouched(
    tmp_path: Path, cache_dir: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"input_model": "nowhere.onnx"}))

    staged_tokens = input_staging.stage_inputs(
        ["--path", str(config_path)], {"--path"}
    )

    rewritten = yaml.safe_load(
        _host_staged_path(staged_tokens[1], cache_dir).read_text()
    )
    assert rewritten["input_model"] == "nowhere.onnx"


def test_non_path_fields_are_never_staged(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    # A stage named after a directory that exists next to the config.
    (tmp_path / "resnet18").mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"name": "resnet18", "stages": {"resnet18": {"input_model": None}}}
        )
    )
    monkeypatch.chdir(tmp_path)

    staged_tokens = input_staging.stage_inputs(
        ["--path", str(config_path)], {"--path"}
    )

    rewritten = yaml.safe_load(
        _host_staged_path(staged_tokens[1], cache_dir).read_text()
    )
    assert rewritten["name"] == "resnet18"


def test_a_config_naming_itself_terminates(
    tmp_path: Path, cache_dir: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"input_model": "config.yaml"}))

    staged_tokens = input_staging.stage_inputs(
        ["--path", str(config_path)], {"--path"}
    )

    rewritten = yaml.safe_load(
        _host_staged_path(staged_tokens[1], cache_dir).read_text()
    )
    assert _staged_content(rewritten["input_model"], cache_dir) == (
        config_path.read_bytes()
    )


def test_restaging_a_changed_reference_keeps_the_previous_config(
    tmp_path: Path, cache_dir: Path
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"first")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"input_model": "model.onnx"}))

    first = input_staging.stage_inputs(
        ["--path", str(config_path)], {"--path"}
    )
    model.write_bytes(b"second")
    second = input_staging.stage_inputs(
        ["--path", str(config_path)], {"--path"}
    )

    # A published entry is keyed by its own content, so the copy a concurrent
    # run may still be reading is never rewritten in place.
    assert first[1] != second[1]
    assert (
        _staged_content(
            yaml.safe_load(_host_staged_path(first[1], cache_dir).read_text())[
                "input_model"
            ],
            cache_dir,
        )
        == b"first"
    )


def test_output_destinations_are_left_alone(
    tmp_path: Path, cache_dir: Path
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    destination = tmp_path / "nas"
    destination.mkdir()
    model = tmp_path / "models" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"model")

    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input_model": str(model),
                "output_remote_url": str(destination),
                "intermediate_outputs_remote_url": str(destination),
            }
        )
    )

    staged_tokens = input_staging.stage_inputs(
        ["--config", str(config_path)], {"--config"}
    )

    rewritten = yaml.safe_load(
        _host_staged_path(staged_tokens[1], cache_dir).read_text()
    )
    assert rewritten["input_model"].startswith(
        f"{CONTAINER_SHARED_DIR}/inputs/"
    )
    assert rewritten["output_remote_url"] == str(destination)
    assert rewritten["intermediate_outputs_remote_url"] == str(destination)


def test_bare_names_are_not_mistaken_for_paths(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # Both a config-override value and the subcommand itself collide with a
    # directory in the working directory.
    (tmp_path / "resnet18").mkdir()
    (tmp_path / "convert").mkdir()
    tokens = ["convert", "rvc4", "name", "resnet18"]

    assert input_staging.stage_inputs(tokens, {"--path"}) == tokens
    assert not cache_dir.exists()


def test_relative_paths_are_still_staged(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "image.jpg").write_bytes(b"image")

    staged_tokens = input_staging.stage_inputs(
        ["convert", "rvc4", "calibration.path", "./calibration"], {"--path"}
    )

    assert staged_tokens[3].startswith(f"{CONTAINER_SHARED_DIR}/inputs/")


def test_staging_a_directory_holding_the_cache_does_not_recurse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cache_dir = home / ".cache" / "modelconverter"
    # An earlier run already populated the cache that now sits inside the
    # directory being staged.
    (cache_dir / "inputs" / "old").mkdir(parents=True)
    (cache_dir / "inputs" / "old" / "junk.bin").write_bytes(b"junk")
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)
    (home / "image.jpg").write_bytes(b"image")

    staged_tokens = input_staging.stage_inputs(
        ["--input-path", str(home)], {"--input-path"}
    )

    staged_dir = _host_staged_path(staged_tokens[1], cache_dir)
    assert (staged_dir / "image.jpg").read_bytes() == b"image"
    assert not (staged_dir / ".cache").exists()


def test_unusable_files_do_not_abort_staging(
    tmp_path: Path, cache_dir: Path
) -> None:
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "image.jpg").write_bytes(b"image")
    (calibration / "dangling").symlink_to(tmp_path / "nowhere")

    staged_tokens = input_staging.stage_inputs(
        ["--input-path", str(calibration)], {"--input-path"}
    )

    staged_dir = _host_staged_path(staged_tokens[1], cache_dir)
    assert (staged_dir / "image.jpg").read_bytes() == b"image"
    assert not (staged_dir / "dangling").exists()


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root can read a file whatever its mode",
)
def test_an_unreadable_file_is_staged_once_it_becomes_readable(
    tmp_path: Path,
) -> None:
    src = tmp_path / "calibration"
    src.mkdir()
    (src / "image.jpg").write_bytes(b"image")
    unreadable = src / "root_owned.jpg"
    unreadable.write_bytes(b"leftover")
    unreadable.chmod(0o000)
    inputs_dir = tmp_path / "cache" / "inputs"

    try:
        without = input_staging._stage_dir(src, inputs_dir)
        assert (without / "image.jpg").read_bytes() == b"image"
        assert not (without / "root_owned.jpg").exists()

        unreadable.chmod(0o644)
        with_it = input_staging._stage_dir(src, inputs_dir)
    finally:
        unreadable.chmod(0o644)

    # The digest describes what the entry holds, so the file joining the
    # enumeration yields a new entry instead of reusing the one without it.
    assert with_it != without
    assert (with_it / "root_owned.jpg").read_bytes() == b"leftover"


def test_a_file_lost_mid_copy_does_not_publish_an_incomplete_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "calibration"
    src.mkdir()
    (src / "one.jpg").write_bytes(b"one")
    (src / "two.jpg").write_bytes(b"two")
    inputs_dir = tmp_path / "cache" / "inputs"
    expected = inputs_dir / input_staging._hash_dir(src) / src.name
    real_copy = shutil.copy2

    def fail_on_second(source: Path, dest: Path, *args, **kwargs) -> None:
        if Path(source).name == "two.jpg":
            raise PermissionError(errno.EACCES, "Permission denied")
        real_copy(source, dest, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", fail_on_second)

    with pytest.raises(PermissionError):
        input_staging._stage_dir(src, inputs_dir)

    assert not expected.exists()


def test_directory_digest_covers_symlinked_subdirectories(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "image.jpg").write_bytes(b"first")
    src = tmp_path / "configs"
    src.mkdir()
    (src / "data").symlink_to(data, target_is_directory=True)

    before = input_staging._hash_dir(src)
    (data / "image.jpg").write_bytes(b"second version")
    after = input_staging._hash_dir(src)

    assert before != after


def _release_claims(entry: Path) -> None:
    """Stands in for the run that staged ``entry`` having exited."""
    for marker in entry.glob(f"{input_staging._IN_USE_PREFIX}*"):
        marker.unlink()


def test_restaging_a_changed_file_replaces_the_previous_copy(
    tmp_path: Path, cache_dir: Path
) -> None:
    """A file entry is superseded the same way a directory one is.

    Without this the cache keeps every version of a model that is
    re-converted while it is being worked on.
    """
    inputs_dir = cache_dir / "inputs"
    src = tmp_path / "model.tflite"
    src.write_bytes(b"first")

    first = input_staging._stage_file(src, inputs_dir)
    _release_claims(first.parent)

    src.write_bytes(b"second version")
    second = input_staging._stage_file(src, inputs_dir)

    assert first != second
    assert not first.exists()
    assert second.read_bytes() == b"second version"


def test_restaging_a_changed_model_replaces_the_previous_bundle(
    tmp_path: Path, cache_dir: Path
) -> None:
    """An ONNX model and its external data are one entry, superseded as
    one."""
    inputs_dir = cache_dir / "inputs"
    model_path = tmp_path / "models" / "model.onnx"
    model_path.parent.mkdir()
    external_data = _external_onnx(model_path)

    first = input_staging._stage_onnx(model_path, inputs_dir)
    _release_claims(first.parent)

    # The weights changed, not the model file that names them.
    external_data.write_bytes(external_data.read_bytes() + b"retrained")
    second = input_staging._stage_onnx(model_path, inputs_dir)

    assert first != second
    assert not first.parent.exists()
    assert second.exists()


def test_restaging_a_changed_config_replaces_the_previous_copy(
    tmp_path: Path, cache_dir: Path
) -> None:
    """The rewritten config is keyed by its own text, so every edit used
    to leave the previous one behind for good."""
    inputs_dir = cache_dir / "inputs"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"input_model": "model.onnx"}))

    first = input_staging._stage_config(config_path, inputs_dir)
    _release_claims(first.parent)

    config_path.write_text(
        yaml.safe_dump({"input_model": "model.onnx", "name": "renamed"})
    )
    second = input_staging._stage_config(config_path, inputs_dir)

    assert first != second
    assert not first.exists()
    assert yaml.safe_load(second.read_text())["name"] == "renamed"


def test_restaging_a_changed_directory_replaces_the_previous_copy(
    tmp_path: Path, cache_dir: Path
) -> None:
    inputs_dir = cache_dir / "inputs"
    src = tmp_path / "calibration"
    src.mkdir()
    (src / "image.jpg").write_bytes(b"first")

    first = input_staging._stage_dir(src, inputs_dir)
    # The run that staged the first copy has exited, dropping its claim.
    _release_claims(first.parent)

    (src / "image.jpg").write_bytes(b"second version")
    second = input_staging._stage_dir(src, inputs_dir)

    assert first != second
    assert not first.exists()
    assert (second / "image.jpg").read_bytes() == b"second version"
    assert [entry.name for entry in inputs_dir.iterdir()] == [
        second.parent.name
    ]


def test_staged_copies_still_in_use_are_not_pruned(
    tmp_path: Path, cache_dir: Path
) -> None:
    inputs_dir = cache_dir / "inputs"
    src = tmp_path / "calibration"
    src.mkdir()
    (src / "image.jpg").write_bytes(b"first")

    first = input_staging._stage_dir(src, inputs_dir)
    # Stand in for a container still reading the first copy: this process
    # claimed it, and `_stage_dir` registers that claim under its own pid.
    assert list(first.parent.glob(f"{input_staging._IN_USE_PREFIX}*"))

    (src / "image.jpg").write_bytes(b"second version")
    second = input_staging._stage_dir(src, inputs_dir)

    assert first != second
    assert (first / "image.jpg").read_bytes() == b"first"
    assert (second / "image.jpg").read_bytes() == b"second version"


def test_staged_copies_claimed_by_dead_processes_are_pruned(
    tmp_path: Path, cache_dir: Path
) -> None:
    inputs_dir = cache_dir / "inputs"
    src = tmp_path / "calibration"
    src.mkdir()
    (src / "image.jpg").write_bytes(b"first")

    first = input_staging._stage_dir(src, inputs_dir)
    # Re-stamp the claim with a pid that is guaranteed not to be running.
    dead_pid = subprocess.Popen([sys.executable, "-c", ""])
    dead_pid.wait()
    for marker in first.parent.glob(f"{input_staging._IN_USE_PREFIX}*"):
        marker.unlink()
    (first.parent / f"{input_staging._IN_USE_PREFIX}{dead_pid.pid}").touch()

    (src / "image.jpg").write_bytes(b"second version")
    second = input_staging._stage_dir(src, inputs_dir)

    assert first != second
    assert not first.exists()
    assert (second / "image.jpg").read_bytes() == b"second version"


def test_path_flags_come_from_the_command_signature() -> None:
    from modelconverter.__main__ import convert, infer

    assert input_staging.path_flags_for(convert) == {"--path"}
    assert input_staging.path_flags_for(infer) == {
        "--config",
        "--input-path",
        "--input_path",
        "--model-path",
        "--model_path",
        "--path",
    }


def test_stages_the_underscored_spelling_of_a_path_flag(
    tmp_path: Path, cache_dir: Path
) -> None:
    """Cyclopts accepts an option spelled either way, so a path given as
    ``--model_path`` must be staged like one given as ``--model-path``.

    Left unstaged it would reach the container as a host path that only
    exists outside it.
    """
    from modelconverter.__main__ import infer

    model_path = tmp_path / "model.dlc"
    model_path.write_bytes(b"model")

    staged_tokens = input_staging.stage_inputs(
        ["--model_path", str(model_path)], input_staging.path_flags_for(infer)
    )

    assert staged_tokens[1] != str(model_path)
    assert _host_staged_path(staged_tokens[1], cache_dir).read_bytes() == (
        model_path.read_bytes()
    )


def test_interrupted_file_copy_does_not_publish_partial_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "model.tflite"
    src.write_bytes(b"complete model")
    inputs_dir = tmp_path / "cache" / "inputs"
    expected = inputs_dir / input_staging._hash_file(src) / src.name

    def interrupt_copy(source: Path, dest: Path) -> None:
        Path(dest).write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "copy2", interrupt_copy)

    with pytest.raises(KeyboardInterrupt):
        input_staging._stage_file(src, inputs_dir)

    assert not expected.exists()
    assert not list(expected.parent.glob(f".{src.name}.tmp-*"))


def test_interrupted_ir_pair_copy_does_not_publish_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = tmp_path / "model.xml"
    bin_path = tmp_path / "model.bin"
    xml.write_bytes(b"xml")
    bin_path.write_bytes(b"weights")
    inputs_dir = tmp_path / "cache" / "inputs"
    expected_dir = inputs_dir / input_staging._hash_files([xml, bin_path])
    real_copy = shutil.copy2
    copy_count = 0

    def interrupt_second_copy(source: Path, dest: Path) -> None:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            Path(dest).write_bytes(b"partial")
            raise KeyboardInterrupt
        real_copy(source, dest)

    monkeypatch.setattr(shutil, "copy2", interrupt_second_copy)

    with pytest.raises(KeyboardInterrupt):
        input_staging._stage_ir_pair(xml, inputs_dir)

    assert not expected_dir.exists()
    assert not list(inputs_dir.glob(f".{expected_dir.name}.tmp-*"))


def test_interrupted_directory_copy_does_not_publish_partial_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "calibration"
    src.mkdir()
    (src / "image.jpg").write_bytes(b"complete image")
    inputs_dir = tmp_path / "cache" / "inputs"
    expected = inputs_dir / input_staging._hash_dir(src) / src.name

    def interrupt_copy(
        files: list[input_staging._InputFile], dest: Path
    ) -> None:
        dest.mkdir(parents=True)
        (dest / "image.jpg").write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(input_staging, "_copy_input_files", interrupt_copy)

    with pytest.raises(KeyboardInterrupt):
        input_staging._stage_dir(src, inputs_dir)

    assert not expected.exists()
    assert not list(expected.parent.glob(f".{src.name}.tmp-*"))


def test_concurrent_directory_staging_publishes_one_complete_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "calibration"
    src.mkdir()
    (src / "one.jpg").write_bytes(b"one")
    (src / "two.jpg").write_bytes(b"two")
    inputs_dir = tmp_path / "cache" / "inputs"
    barrier = threading.Barrier(2)
    real_copy = input_staging._copy_input_files

    def synchronized_copy(
        files: list[input_staging._InputFile], dest: Path
    ) -> None:
        barrier.wait(timeout=5)
        real_copy(files, dest)

    monkeypatch.setattr(input_staging, "_copy_input_files", synchronized_copy)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: input_staging._stage_dir(src, inputs_dir),
                range(2),
            )
        )

    assert results[0] == results[1]
    assert (results[0] / "one.jpg").read_bytes() == b"one"
    assert (results[0] / "two.jpg").read_bytes() == b"two"
    assert not list(results[0].parent.glob(f".{src.name}.tmp-*"))


def test_a_symlinked_model_keeps_the_name_it_was_given(
    tmp_path: Path, cache_dir: Path
) -> None:
    """DVC, git-annex and Nix all present a model as a symlink to a
    content-addressed blob.

    The container dispatches on the suffix, so a staged copy named after
    the link's target -- extension-less -- is parsed as a config instead
    of converted.
    """
    store = tmp_path / "store"
    store.mkdir()
    blob = store / "b1946ac92492d234"
    single_io_onnx(store / "model.onnx").rename(blob)
    work = tmp_path / "work"
    work.mkdir()
    link = work / "model.onnx"
    link.symlink_to(blob)

    staged = input_staging.stage_inputs(
        ["--model-path", str(link)], {"--model-path"}
    )[1]

    assert Path(staged).name == "model.onnx"
    assert (
        _host_staged_path(staged, cache_dir).read_bytes() == blob.read_bytes()
    )


def test_a_long_override_value_is_not_treated_as_a_path(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rvc4.encodings` accepts the encodings JSON inline.

    It has no separator, so it reaches the filesystem as one component
    and `Path.exists` answers with `ENAMETOOLONG` rather than `False`.
    """
    monkeypatch.chdir(tmp_path)
    encodings = (
        '{"activation_encodings": {"x": ['
        + '{"bitwidth": 8},' * 40
        + '{"bitwidth": 8}]}, "param_encodings": {}}'
    )
    assert len(encodings) > 255
    tokens = ["convert", "rvc4", "rvc4.encodings", encodings]

    assert input_staging.stage_inputs(tokens, {"--path"}) == tokens


def test_stages_the_quantization_overrides_file(
    tmp_path: Path, cache_dir: Path
) -> None:
    """`rvc4.snpe_onnx_to_dlc_args` is a raw argument list, but the value
    after `--quantization_overrides` is a file the config opens."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    overrides = config_dir / "encodings.json"
    overrides.write_text('{"activation_encodings": {}, "param_encodings": {}}')
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "rvc4": {
                    "snpe_onnx_to_dlc_args": [
                        "--quantization_overrides",
                        "encodings.json",
                        "--float_bw",
                        "16",
                    ]
                }
            }
        )
    )

    staged = input_staging.stage_inputs(
        ["--config", str(config_path)], {"--config"}
    )[1]

    rewritten = yaml.safe_load(
        _host_staged_path(staged, cache_dir).read_text()
    )
    args = rewritten["rvc4"]["snpe_onnx_to_dlc_args"]
    assert args[0] == "--quantization_overrides"
    assert args[1].startswith(f"{CONTAINER_SHARED_DIR}/inputs/")
    # Everything else in the list is passed to the tool verbatim.
    assert args[2:] == ["--float_bw", "16"]
    assert (
        _host_staged_path(args[1], cache_dir).read_text()
        == overrides.read_text()
    )


def test_stages_the_transformations_config(
    tmp_path: Path, cache_dir: Path
) -> None:
    """`--transformations_config` names a file the model optimizer opens,
    so it is staged like any other input the container has to read."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    transformations = config_dir / "custom.json"
    transformations.write_text("[]")
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "rvc2": {
                    "mo_args": [
                        "--transformations_config",
                        "custom.json",
                        "--static_shape",
                    ]
                }
            }
        )
    )

    staged = input_staging.stage_inputs(
        ["--config", str(config_path)], {"--config"}
    )[1]

    rewritten = yaml.safe_load(
        _host_staged_path(staged, cache_dir).read_text()
    )
    args = rewritten["rvc2"]["mo_args"]
    assert args[0] == "--transformations_config"
    assert args[1].startswith(f"{CONTAINER_SHARED_DIR}/inputs/")
    assert args[2:] == ["--static_shape"]
    assert (
        _host_staged_path(args[1], cache_dir).read_text()
        == transformations.read_text()
    )


def test_stages_the_compile_tool_config(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`compile_tool -c` names a configuration file, which the exporter
    writes itself only when the user has not supplied one."""
    monkeypatch.chdir(tmp_path)
    compile_config = tmp_path / "compile.conf"
    compile_config.write_text("MYRIAD_NUMBER_OF_SHAVES 4")

    staged = input_staging.stage_inputs(
        [
            "convert",
            "rvc2",
            "rvc2.compile_tool_args",
            '["-c", "compile.conf", "-ip", "U8"]',
        ],
        {"--path"},
    )[3]

    args = ast.literal_eval(staged)
    assert args[0] == "-c"
    assert args[1].startswith(f"{CONTAINER_SHARED_DIR}/inputs/")
    assert args[2:] == ["-ip", "U8"]
    assert (
        _host_staged_path(args[1], cache_dir).read_text()
        == compile_config.read_text()
    )


def test_sibling_symlinks_to_one_directory_are_both_staged(
    tmp_path: Path, cache_dir: Path
) -> None:
    """The inferer expects one directory per input, and two of them may
    well be the same images under different names."""
    shared = tmp_path / "images"
    shared.mkdir()
    (shared / "image.raw").write_bytes(b"image")
    data = tmp_path / "data"
    data.mkdir()
    (data / "input0").symlink_to(shared, target_is_directory=True)
    (data / "input1").symlink_to(shared, target_is_directory=True)

    staged = input_staging.stage_inputs(
        ["--input-path", str(data)], {"--input-path"}
    )[1]

    staged_dir = _host_staged_path(staged, cache_dir)
    assert (staged_dir / "input0" / "image.raw").read_bytes() == b"image"
    assert (staged_dir / "input1" / "image.raw").read_bytes() == b"image"


def test_a_symlink_pointing_back_up_is_not_followed(
    tmp_path: Path, cache_dir: Path
) -> None:
    """A convenience link such as `calib/all -> ..` would otherwise copy
    the whole surrounding tree into the cache."""
    (tmp_path / "unrelated.bin").write_bytes(b"unrelated")
    calib = tmp_path / "calib"
    calib.mkdir()
    (calib / "image.raw").write_bytes(b"image")
    (calib / "all").symlink_to(tmp_path, target_is_directory=True)
    (calib / "here").symlink_to(calib, target_is_directory=True)

    staged = input_staging.stage_inputs(
        ["--input-path", str(calib)], {"--input-path"}
    )[1]

    staged_dir = _host_staged_path(staged, cache_dir)
    assert (staged_dir / "image.raw").read_bytes() == b"image"
    assert not (staged_dir / "all").exists()
    assert not (staged_dir / "here").exists()


def test_an_output_destination_override_is_left_alone(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rewriting a destination would upload the converted model into the
    cache instead of where the user asked for it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nas").mkdir()
    tokens = ["convert", "rvc4", "output_remote_url", "./nas"]

    assert input_staging.stage_inputs(tokens, {"--path"}) == tokens


def test_a_bare_directory_name_is_staged_for_a_path_override(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config schema says the value is a path, which beats guessing
    from a shape that a bare name does not have."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "calib").mkdir()
    (tmp_path / "calib" / "image.raw").write_bytes(b"image")
    tokens = ["convert", "rvc4", "calibration.path", "calib"]

    staged = input_staging.stage_inputs(tokens, {"--path"})

    assert staged[3].startswith(f"{CONTAINER_SHARED_DIR}/inputs/")
    assert (
        _host_staged_path(staged[3], cache_dir) / "image.raw"
    ).read_bytes() == b"image"


def test_a_single_dash_flag_value_is_not_staged(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-c` takes a shell command, not an input path; only long options
    were recognised as introducing a value."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "script.py").write_bytes(b"print()")
    tokens = ["shell", "rvc4", "-c", "./script.py"]

    assert input_staging.stage_inputs(tokens, {"--path"}) == tokens


def test_replacing_directory_content_in_place_restages_it(
    tmp_path: Path, cache_dir: Path
) -> None:
    """`rsync -a`, `cp -p` and a backup restore all preserve size and
    modification time.

    Keying the cache on those alone would hand the container the previous
    calibration images and quantize against them without a word.
    """
    calib = tmp_path / "calib"
    calib.mkdir()
    image = calib / "image.raw"
    image.write_bytes(b"aaaa")

    first = input_staging.stage_inputs(
        ["--input-path", str(calib)], {"--input-path"}
    )[1]

    original = image.stat()
    image.write_bytes(b"bbbb")
    os.utime(image, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert image.stat().st_size == original.st_size
    assert image.stat().st_mtime_ns == original.st_mtime_ns

    second = input_staging.stage_inputs(
        ["--input-path", str(calib)], {"--input-path"}
    )[1]

    assert second != first
    assert (
        _host_staged_path(second, cache_dir) / "image.raw"
    ).read_bytes() == b"bbbb"


def _fabricated_file(
    path: Path, **stat_fields: int
) -> "input_staging._InputFile":
    """An enumerated file with a stat this test controls.

    Two filesystems cannot be mounted from a unit test, so the one
    property that distinguishes them -- `st_dev` -- has to be supplied.
    """
    stat = SimpleNamespace(
        st_size=4, st_mtime_ns=1, st_ctime_ns=1, st_dev=1, st_ino=7
    )
    for name, value in stat_fields.items():
        setattr(stat, name, value)
    return input_staging._InputFile("sample.npy", path, stat)  # type: ignore[arg-type]


def test_directory_digest_separates_matching_inodes_on_other_devices(
    tmp_path: Path,
) -> None:
    """An inode number is only unique within one filesystem.

    Two calibration sets on different mounts can agree on inode, size
    and modification time, and everything else in the digest is the file
    name -- so without the device they share a cache entry and the
    second conversion silently reuses the first one's images.
    """
    on_one = _fabricated_file(tmp_path / "a" / "sample.npy", st_dev=1)
    on_other = _fabricated_file(tmp_path / "b" / "sample.npy", st_dev=2)

    assert input_staging._hash_dir(Path("data"), [on_one]) != (
        input_staging._hash_dir(Path("data"), [on_other])
    )


def test_a_serialized_argument_list_override_is_staged(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same list written in a config file is rewritten already.

    On the command line it arrives as one token, so nothing saw the path
    inside it -- and the container opened it during config validation,
    where the host path does not exist.
    """
    monkeypatch.chdir(tmp_path)
    overrides = tmp_path / "encodings.json"
    overrides.write_text('{"activation_encodings": {}, "param_encodings": {}}')

    staged = input_staging.stage_inputs(
        [
            "convert",
            "rvc4",
            "rvc4.snpe_onnx_to_dlc_args",
            '["--quantization_overrides", "encodings.json", "--float_bw", "16"]',
        ],
        {"--path"},
    )[3]

    # `LuxonisConfig` parses an override value with `ast.literal_eval`, so the
    # rewritten token has to survive the same round trip.
    args = ast.literal_eval(staged)
    assert args[0] == "--quantization_overrides"
    assert args[1].startswith(f"{CONTAINER_SHARED_DIR}/inputs/")
    assert args[2:] == ["--float_bw", "16"]
    assert (
        _host_staged_path(args[1], cache_dir).read_text()
        == overrides.read_text()
    )


@pytest.mark.parametrize(
    "value",
    [
        "not a list at all",
        '["--float_bw", "16"]',
        '["--quantization_overrides", "missing.json"]',
    ],
)
def test_argument_list_overrides_without_a_local_path_are_untouched(
    tmp_path: Path,
    cache_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """Only the value after a path flag is ours to rewrite.

    A missing file is left for the config validation to report, the same
    way a missing reference inside a config file is.
    """
    monkeypatch.chdir(tmp_path)
    tokens = ["convert", "rvc4", "rvc4.snpe_onnx_to_dlc_args", value]

    assert input_staging.stage_inputs(tokens, {"--path"}) == tokens


def test_a_dotted_output_destination_override_is_left_alone(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A destination field keeps naming a destination when it is reached
    through a stage prefix."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "exported").mkdir()
    tokens = [
        "convert",
        "rvc4",
        "stages.model.output_remote_url",
        "./exported",
        "intermediate_outputs_remote_url",
        "./exported",
    ]

    assert input_staging.stage_inputs(tokens, {"--path"}) == tokens


def test_stages_the_flag_equals_value_quantization_overrides(
    tmp_path: Path, cache_dir: Path
) -> None:
    """`--quantization_overrides=file` names the same file as the
    two-token spelling and has to reach the container all the same."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    overrides = config_dir / "encodings.json"
    overrides.write_text('{"activation_encodings": {}, "param_encodings": {}}')
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "rvc4": {
                    "snpe_onnx_to_dlc_args": [
                        "--quantization_overrides=encodings.json",
                        "--float_bw",
                        "16",
                    ]
                }
            }
        )
    )

    staged = input_staging.stage_inputs(
        ["--config", str(config_path)], {"--config"}
    )[1]

    rewritten = yaml.safe_load(
        _host_staged_path(staged, cache_dir).read_text()
    )
    args = rewritten["rvc4"]["snpe_onnx_to_dlc_args"]
    flag, _, value = args[0].partition("=")
    assert flag == "--quantization_overrides"
    assert value.startswith(f"{CONTAINER_SHARED_DIR}/inputs/")
    assert args[1:] == ["--float_bw", "16"]
    assert (
        _host_staged_path(value, cache_dir).read_text()
        == overrides.read_text()
    )


def test_a_negative_decimal_is_not_mistaken_for_a_flag(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-.5` is an override value; treating it as an option would exempt
    whatever comes after it from staging."""
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model.onnx"
    single_io_onnx(model)

    staged = input_staging.stage_inputs(
        ["convert", "rvc4", "outputs.0.scale", "-.5", "./model.onnx"],
        {"--path"},
    )

    assert staged[3] == "-.5"
    assert staged[4].startswith(f"{CONTAINER_SHARED_DIR}/inputs/")


def test_staged_files_are_claimed_while_the_process_lives(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cache clean` consults the claims, so a conversion whose staged
    inputs are plain files needs them no less than one that staged a
    directory."""
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model.onnx"
    single_io_onnx(model)

    staged = input_staging.stage_inputs(["--path", str(model)], {"--path"})[1]

    entry = _host_staged_path(staged, cache_dir).parent
    assert entry in input_staging.in_use_staged_inputs()


def test_a_claim_from_a_recycled_pid_is_ignored(tmp_path: Path) -> None:
    """A killed run leaves its marker behind, and its pid may since have
    been handed to an unrelated long-lived process; the recorded creation
    time is what tells the two apart."""
    entry = tmp_path / "digest"
    entry.mkdir()
    marker = entry / f"{input_staging._IN_USE_PREFIX}{os.getpid()}"
    # This pid is certainly alive, but the claim predates its process.
    marker.write_text("1.0")

    assert not input_staging._is_in_use(entry)
    assert not marker.exists()


def test_path_flags_of_no_command() -> None:
    """The launcher parses the command before staging; when it has none
    there is nothing whose signature could name a path."""
    assert input_staging.path_flags_for(None) == set()


def test_path_flags_of_an_uninspectable_command() -> None:
    """Staging follows the command signature, so a command it cannot
    read leaves every token alone rather than guessing."""
    assert input_staging.path_flags_for(object()) == set()  # type: ignore[arg-type]


def test_stages_a_path_joined_to_its_flag(
    tmp_path: Path, cache_dir: Path
) -> None:
    """`--model-path=<path>` carries the path in the same token as the
    flag, so the token has to be rewritten around it."""
    model = tmp_path / "model.dlc"
    model.write_bytes(b"dlc")

    (token,) = input_staging.stage_inputs(
        [f"--model-path={model}"], {"--model-path"}
    )

    flag, _, staged = token.partition("=")
    assert flag == "--model-path"
    assert _host_staged_path(staged, cache_dir).read_bytes() == b"dlc"


def test_leaves_a_joined_value_of_another_flag_alone(cache_dir: Path) -> None:
    assert input_staging.stage_inputs(
        ["--to=nn_archive"], {"--model-path"}
    ) == ["--to=nn_archive"]


def test_arg_list_override_that_is_not_a_list(tmp_path: Path) -> None:
    """The override is read the way `LuxonisConfig` reads it, and a
    value that is not a list holds no arguments to rewrite."""
    assert (
        input_staging._stage_arg_list_token(
            "42", "snpe_onnx_to_dlc_args", tmp_path
        )
        is None
    )


def test_a_remote_url_is_not_path_like() -> None:
    assert input_staging._is_path_like("s3://bucket/model.onnx") is False


def test_a_dot_prefixed_name_is_path_like(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leading `.` says the user meant a path, so the name is staged
    even without a known extension."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".hidden").write_text("x")

    assert input_staging._is_path_like(".hidden") is True


def test_a_value_naming_no_home_is_not_a_path() -> None:
    """`~` for a user that does not exist is not something `Path` can
    represent."""
    assert input_staging._as_path("~nosuchuser4242/model.onnx") is None


def test_a_remote_url_is_never_staged(tmp_path: Path) -> None:
    assert input_staging._stage_value("s3://bucket/m.onnx", tmp_path) is None


def test_publishing_reraises_an_unexpected_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing the rename race is normal and accepted; anything else is
    not staging's to swallow."""

    def refuse_rename(self: Path, target: Path) -> Path:
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr(Path, "rename", refuse_rename)

    with pytest.raises(OSError, match="denied"):
        input_staging._publish_directory(tmp_path / "src", tmp_path / "dest")


def test_an_unreadable_config_is_staged_verbatim(
    tmp_path: Path, cache_dir: Path
) -> None:
    """Staging must not be what reports a broken config: the container
    validates it and says something useful."""
    config = tmp_path / "broken.yaml"
    config.write_text("a: [1, 2\n")

    (staged,) = input_staging.stage_inputs([str(config)], set())

    assert _host_staged_path(staged, cache_dir).read_text() == "a: [1, 2\n"


def test_a_config_reference_to_a_remote_url_is_left_alone(
    tmp_path: Path,
) -> None:
    assert (
        input_staging._stage_config_reference(
            "s3://bucket/model.onnx", tmp_path, tmp_path
        )
        is None
    )


def test_a_config_reference_that_is_not_a_path_is_left_alone(
    tmp_path: Path,
) -> None:
    """`calibration.script` may hold the script itself rather than a
    path to one."""
    assert (
        input_staging._stage_config_reference(
            "~nosuchuser4242/x", tmp_path, tmp_path
        )
        is None
    )


def test_staging_a_directory_skips_metadata_and_special_files(
    tmp_path: Path, cache_dir: Path
) -> None:
    """Repository metadata is not a model input and would dominate the
    copy; anything that is not a regular file cannot be copied at all.
    """
    src = tmp_path / "calibration"
    (src / ".git").mkdir(parents=True)
    (src / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (src / "image.npy").write_bytes(b"data")
    os.mkfifo(src / "pipe")

    (staged,) = input_staging.stage_inputs([str(src)], set())

    staged_dir = _host_staged_path(staged, cache_dir)
    assert (staged_dir / "image.npy").read_bytes() == b"data"
    assert not (staged_dir / ".git").exists()
    assert not (staged_dir / "pipe").exists()


def _break_stat(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Makes ``Path.stat`` fail for the file called ``name``."""
    real_stat = Path.stat

    def failing_stat(self: Path, **kwargs: bool) -> os.stat_result:
        if self.name == name:
            raise OSError("stat failed")
        # Guards anything else that stats while the patch is in place.
        return real_stat(self, **kwargs)  # pragma: no cover

    monkeypatch.setattr(Path, "stat", failing_stat)


def test_a_directory_that_cannot_be_stat_ed_has_no_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The identity is what a symlink cycle repeats; without one the
    directory is simply skipped."""
    _break_stat(monkeypatch, "unstattable")

    assert input_staging._dir_key(tmp_path / "unstattable") is None


def test_a_claim_that_cannot_be_written_is_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only cache still has to let the conversion run; the claim
    is an optimisation, not a precondition."""

    def refuse_write(dest: Path, content: str) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(input_staging, "_atomic_write_text", refuse_write)

    input_staging._mark_in_use(tmp_path)

    assert not input_staging._is_in_use(tmp_path)


def test_an_entry_whose_claims_cannot_be_listed_is_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A downloaded file is not a directory and holds no claims."""

    def refuse_glob(self: Path, pattern: str) -> Iterator[Path]:
        raise OSError("not a directory")

    monkeypatch.setattr(Path, "glob", refuse_glob)

    assert input_staging._is_in_use(tmp_path) is False


def test_a_claim_without_a_pid_is_ignored(tmp_path: Path) -> None:
    (tmp_path / f"{input_staging._IN_USE_PREFIX}notapid").write_text("0")

    assert input_staging._is_in_use(tmp_path) is False


def test_a_claim_that_cannot_be_inspected_still_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pid exists but belongs to another user, so the entry may
    still be in use and must not be evicted."""
    marker = tmp_path / f"{input_staging._IN_USE_PREFIX}{os.getpid()}"
    marker.write_text("0.0")

    def refuse_inspection(pid: int) -> psutil.Process:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(input_staging.psutil, "Process", refuse_inspection)

    assert input_staging._is_in_use(tmp_path) is True


def test_claiming_an_unwritable_cache_is_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache)

    def refuse_mkdir(
        self: Path, parents: bool = False, exist_ok: bool = False
    ) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "mkdir", refuse_mkdir)

    input_staging.claim_cache()

    assert not cache.exists()


def test_trash_without_a_pid_in_its_name_is_kept(tmp_path: Path) -> None:
    """The name records the sweep that made it; one that carries no pid
    cannot be shown to be stale."""
    trash = tmp_path / f"{input_staging._TRASH_PREFIX}notapid-entry"
    trash.mkdir()

    input_staging._discard_stale_trash(trash)

    assert trash.exists()


def test_an_uninspectable_process_starts_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The start time only protects what this run staged; falling back
    to now keeps that protection rather than dropping it."""

    def refuse_inspection() -> psutil.Process:
        raise psutil.AccessDenied(0)

    monkeypatch.setattr(input_staging.psutil, "Process", refuse_inspection)

    assert input_staging._process_start_time() > 0


def test_an_entry_that_cannot_be_renamed_is_not_evicted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "entry"
    entry.mkdir()

    def refuse_rename(self: Path, target: Path) -> Path:
        raise OSError("busy")

    monkeypatch.setattr(Path, "rename", refuse_rename)

    assert input_staging._evict(entry) is False
    assert entry.exists()


def test_an_entry_claimed_mid_eviction_that_cannot_be_put_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claim that landed between the check and the rename normally
    puts the entry back. When even that fails the copy is dropped, which
    costs a re-staging rather than leaving it under a trash name."""
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / f"{input_staging._IN_USE_PREFIX}{os.getpid()}").write_text(
        str(psutil.Process().create_time())
    )

    real_rename = Path.rename
    renames: list[Path] = []

    def refuse_second_rename(self: Path, target: Path) -> Path:
        renames.append(self)
        if len(renames) == 1:
            return real_rename(self, target)
        raise OSError("busy")

    monkeypatch.setattr(Path, "rename", refuse_second_rename)

    assert input_staging._evict(entry) is False
    assert not entry.exists()


def test_a_file_that_cannot_be_stat_ed_is_hashed_directly(
    tmp_path: Path, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memo is keyed on the stat fingerprint; without one the
    digest is still a description of the bytes."""
    path = tmp_path / "model.dlc"
    path.write_bytes(b"payload")
    expected = input_staging._hash_file(path)

    _break_stat(monkeypatch, "model.dlc")

    assert input_staging._hash_file_memoized(path) == expected
