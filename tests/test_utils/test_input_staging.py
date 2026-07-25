import shutil
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


def test_stages_onnx_external_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    model_path = tmp_path / "models" / "model.onnx"
    model_path.parent.mkdir()
    external_data = _external_onnx(model_path)
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

    staged_tokens = input_staging.stage_inputs(
        ["--model-path", str(model_path)]
    )

    staged_model = _host_staged_path(staged_tokens[1], cache_dir)
    assert staged_model.read_bytes() == model_path.read_bytes()
    assert staged_model.with_name(external_data.name).read_bytes() == (
        external_data.read_bytes()
    )


def test_stages_and_rewrites_absolute_config_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    relative_model = config_dir / "relative.onnx"
    relative_model.write_bytes(b"relative")

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
                "script": str(script),
                "rvc4": {"encodings": str(encodings)},
                "relative_model": relative_model.name,
                "remote_model": "s3://bucket/model.onnx",
            }
        )
    )
    monkeypatch.setattr(input_staging, "get_cache_dir", lambda: cache_dir)

    staged_tokens = input_staging.stage_inputs(["--config", str(config_path)])

    staged_config = _host_staged_path(staged_tokens[1], cache_dir)
    rewritten = yaml.safe_load(staged_config.read_text())
    for key in ("input_model", "script"):
        assert rewritten[key].startswith(f"{CONTAINER_SHARED_DIR}/inputs/")
    assert rewritten["calibration"]["path"].startswith(
        f"{CONTAINER_SHARED_DIR}/inputs/"
    )
    assert rewritten["rvc4"]["encodings"].startswith(
        f"{CONTAINER_SHARED_DIR}/inputs/"
    )
    assert rewritten["relative_model"] == relative_model.name
    assert rewritten["remote_model"] == "s3://bucket/model.onnx"
    assert (
        staged_config.with_name(relative_model.name).read_bytes()
        == b"relative"
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

    def interrupt_copytree(source: Path, dest: Path) -> None:
        Path(dest).mkdir()
        (Path(dest) / "image.jpg").write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "copytree", interrupt_copytree)

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
    real_copytree = shutil.copytree

    def synchronized_copytree(source: Path, dest: Path) -> Path:
        barrier.wait(timeout=5)
        return real_copytree(source, dest)

    monkeypatch.setattr(shutil, "copytree", synchronized_copytree)

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
