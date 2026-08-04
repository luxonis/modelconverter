import errno
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnx
import pytest
import yaml
from onnx import helper, numpy_helper

from modelconverter.utils import input_staging
from modelconverter.utils.constants import CONTAINER_SHARED_DIR


def _host_staged_path(staged: str, cache_dir: Path) -> Path:
    relative = Path(staged).relative_to(CONTAINER_SHARED_DIR)
    return cache_dir / relative


def _external_onnx(path: Path) -> Path:
    tensor = numpy_helper.from_array(
        np.asarray([1.0, 2.0], dtype=np.float32), name="weights"
    )
    graph = helper.make_graph([], "external", [], [], [tensor])
    model = helper.make_model(graph)
    external_data = path.with_name(f"{path.name}_data")
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data.name,
        size_threshold=0,
    )
    return external_data


def _multi_file_external_onnx(path: Path) -> list[Path]:
    """Saves a model whose tensors each land in their own companion
    file."""
    tensors = [
        numpy_helper.from_array(
            np.asarray([1.0, 2.0], dtype=np.float32), name="first"
        ),
        numpy_helper.from_array(
            np.asarray([3.0, 4.0], dtype=np.float32), name="second"
        ),
    ]
    graph = helper.make_graph([], "external", [], [], tensors)
    model = helper.make_model(graph)
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=0,
    )
    return sorted(p for p in path.parent.iterdir() if p != path)


def test_stages_onnx_external_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    model_path = tmp_path / "models" / "model.onnx"
    model_path.parent.mkdir()
    external_data = _external_onnx(model_path)
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

    staged_tokens = input_staging.stage_inputs(
        ["--model-path", str(model_path)], {"--model-path"}
    )

    staged_model = _host_staged_path(staged_tokens[1], cache_dir)
    assert staged_model.read_bytes() == model_path.read_bytes()
    assert staged_model.with_name(external_data.name).read_bytes() == (
        external_data.read_bytes()
    )


def test_stages_every_external_data_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    model_path = tmp_path / "models" / "model.onnx"
    model_path.parent.mkdir()
    external_data = _multi_file_external_onnx(model_path)
    assert len(external_data) > 1
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
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
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
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
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
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
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

    input_staging.stage_inputs(["--path", str(config_path)], {"--path"})

    staged_names = {
        path.name for path in cache_dir.rglob("*") if path.is_file()
    }
    assert staged_names == {"config.yaml", "model.onnx"}


def test_stages_the_openvino_bin_alongside_the_xml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
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
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"input_model": "nowhere.onnx"}))
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

    staged_tokens = input_staging.stage_inputs(
        ["--path", str(config_path)], {"--path"}
    )

    rewritten = yaml.safe_load(
        _host_staged_path(staged_tokens[1], cache_dir).read_text()
    )
    assert rewritten["input_model"] == "nowhere.onnx"


def test_non_path_fields_are_never_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
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
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)
    monkeypatch.chdir(tmp_path)

    staged_tokens = input_staging.stage_inputs(
        ["--path", str(config_path)], {"--path"}
    )

    rewritten = yaml.safe_load(
        _host_staged_path(staged_tokens[1], cache_dir).read_text()
    )
    assert rewritten["name"] == "resnet18"


def test_a_config_naming_itself_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"input_model": "config.yaml"}))
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"first")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"input_model": "model.onnx"}))
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
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
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)
    monkeypatch.chdir(tmp_path)
    # Both a config-override value and the subcommand itself collide with a
    # directory in the working directory.
    (tmp_path / "resnet18").mkdir()
    (tmp_path / "convert").mkdir()
    tokens = ["convert", "rvc4", "name", "resnet18"]

    assert input_staging.stage_inputs(tokens, {"--path"}) == tokens
    assert not cache_dir.exists()


def test_relative_paths_are_still_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "image.jpg").write_bytes(b"image")
    (calibration / "dangling").symlink_to(tmp_path / "nowhere")
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

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


def test_restaging_a_changed_directory_replaces_the_previous_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    inputs_dir = cache_dir / "inputs"
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)
    src = tmp_path / "calibration"
    src.mkdir()
    (src / "image.jpg").write_bytes(b"first")

    first = input_staging._stage_dir(src, inputs_dir)
    # The run that staged the first copy has exited, dropping its claim.
    for marker in first.parent.glob(f"{input_staging._IN_USE_PREFIX}*"):
        marker.unlink()

    (src / "image.jpg").write_bytes(b"second version")
    second = input_staging._stage_dir(src, inputs_dir)

    assert first != second
    assert not first.exists()
    assert (second / "image.jpg").read_bytes() == b"second version"
    assert [entry.name for entry in inputs_dir.iterdir()] == [
        second.parent.name
    ]


def test_staged_copies_still_in_use_are_not_pruned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    inputs_dir = cache_dir / "inputs"
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    inputs_dir = cache_dir / "inputs"
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cyclopts accepts an option spelled either way, so a path given as
    ``--model_path`` must be staged like one given as ``--model-path``.

    Left unstaged it would reach the container as a host path that only
    exists outside it.
    """
    from modelconverter.__main__ import infer

    cache_dir = tmp_path / "cache"
    model_path = tmp_path / "model.dlc"
    model_path.write_bytes(b"model")
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

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
