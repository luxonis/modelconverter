import json
import tarfile
from pathlib import Path

import pytest
import yaml
from onnx import checker, helper
from onnx.onnx_pb import TensorProto

from modelconverter.utils.config import Config
from modelconverter.utils.docker_utils import generate_compose_config
from modelconverter.utils.telemetry import (
    MODELCONVERTER_TELEMETRY_DEFAULTS,
    build_conversion_result_properties,
    build_conversion_summary,
    detect_config_source,
    get_component_telemetry,
)
from modelconverter.utils.types import Target


def _create_dummy_onnx(path: Path) -> None:
    input_tensor = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    output_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 10]
    )
    node = helper.make_node(
        "Flatten",
        inputs=["input0"],
        outputs=["output0"],
    )
    graph = helper.make_graph(
        [node],
        "DummyModel",
        [input_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph, producer_name="test")
    checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())


def _create_dummy_archive(path: Path, model_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_path = path.parent / "config.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")
    with tarfile.open(path, "w:xz") as tar:
        tar.add(str(model_path), arcname=model_path.name)
        tar.add(str(config_path), arcname="config.json")


def test_build_conversion_summary_rvc4(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy_model.onnx"
    calibration_path = tmp_path / "calibration"
    calibration_path.mkdir()
    _create_dummy_onnx(model_path)

    cfg = Config(
        name="dummy",
        input_model=str(model_path),
        calibration={"path": str(calibration_path)},
        keep_intermediate_outputs=False,
        onnx_simplification=False,
        onnx_optimizations=False,
        output_remote_url="s3://bucket/output",
        intermediate_outputs_remote_url="s3://bucket/intermediate",
        rvc4={
            "optimization_level": 3,
            "quantization_mode": "INT8_STANDARD",
            "use_per_channel_quantization": True,
            "use_per_row_quantization": False,
            "keep_raw_images": True,
            "htp_socs": ["sm8550", "qcs8550"],
        },
    )

    summary = build_conversion_summary(
        cfg,
        target=Target.RVC4,
        config_source="direct_model_input",
        archive_output_mode="nn_archive",
        archive_preprocess=False,
        main_stage_provided=True,
    )

    assert summary == {
        "target": "rvc4",
        "config_source": "direct_model_input",
        "stage_count_bucket": "1",
        "is_multistage": False,
        "main_stage_provided": True,
        "archive_output_mode": "nn_archive",
        "archive_preprocess": False,
        "input_model_format": "onnx",
        "multiple_input_model_formats": False,
        "input_count_bucket": "1",
        "output_count_bucket": "1",
        "calibration_source": "image_directory",
        "disable_calibration": False,
        "keep_intermediate_outputs": False,
        "disable_onnx_simplification": True,
        "disable_onnx_optimization": True,
        "has_remote_output": True,
        "has_remote_intermediate_output": True,
        "target_configuration": {
            "quantization_mode": "int8_standard",
            "optimization_level": 3,
            "use_per_channel_quantization": True,
            "use_per_row_quantization": False,
            "keep_raw_images": True,
            "htp_soc_count_bucket": "2_4",
            "has_quantization_overrides": False,
        },
    }


def test_detect_config_source(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy_model.onnx"
    archive_path = tmp_path / "archive.tar.xz"
    _create_dummy_onnx(model_path)
    _create_dummy_archive(archive_path, model_path)

    assert detect_config_source("model.onnx", [], None) == "direct_model_input"
    assert detect_config_source("config.yaml", [], None) == "yaml_config"
    assert (
        detect_config_source(None, ["input_model", "model.onnx"], None)
        == "direct_model_input"
    )
    assert (
        detect_config_source(str(archive_path), [], object()) == "nn_archive"
    )
    assert (
        detect_config_source("archive_dir", [], object())
        == "archive_directory"
    )


def test_build_conversion_result_properties():
    properties = build_conversion_result_properties(
        result="failed",
        failure_reason="upload_error",
        duration_ms=1234,
        output_artifact_count=3,
        uploaded_output=True,
        uploaded_intermediate_outputs=False,
        peak_ram_bytes=2 * 1024 * 1024 * 1024,
    )

    assert properties == {
        "result": "failed",
        "failure_reason": "upload_error",
        "duration_ms": 1234,
        "output_artifact_count_bucket": "2_4",
        "uploaded_output": True,
        "uploaded_intermediate_outputs": False,
        "peak_ram_usage_bucket": "1g_4g",
    }


def test_generate_compose_config_includes_extra_environment():
    compose = yaml.safe_load(
        generate_compose_config(
            "luxonis/modelconverter-rvc4:test",
            extra_environment={
                "MODELCONVERTER_CONVERSION_RUN_ID": "run-123",
                "LUXONIS_TELEMETRY_ENABLED": "true",
            },
        )
    )

    environment = compose["services"]["modelconverter"]["environment"]
    assert environment["MODELCONVERTER_CONVERSION_RUN_ID"] == "run-123"
    assert environment["LUXONIS_TELEMETRY_ENABLED"] == "true"


def test_get_component_telemetry_uses_modelconverter_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LUXONIS_TELEMETRY_BACKEND", raising=False)
    monkeypatch.delenv("LUXONIS_TELEMETRY_API_KEY", raising=False)
    monkeypatch.delenv("LUXONIS_TELEMETRY_ENDPOINT", raising=False)
    monkeypatch.delenv("LUXONIS_TELEMETRY_DEBUG", raising=False)

    telemetry = get_component_telemetry()

    assert (
        telemetry._config.backend == MODELCONVERTER_TELEMETRY_DEFAULTS.backend
    )
    assert (
        telemetry._config.api_key == MODELCONVERTER_TELEMETRY_DEFAULTS.api_key
    )
    assert (
        telemetry._config.endpoint
        == MODELCONVERTER_TELEMETRY_DEFAULTS.endpoint
    )
