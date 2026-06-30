import os
import resource
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from luxonis_ml.nn_archive import is_nn_archive
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.telemetry import (
    Telemetry,
    TelemetryConfig,
    TelemetryDefaults,
    get_or_init,
    system_context_provider,
)

from modelconverter import __version__
from modelconverter.utils.config import (
    Config,
    HailoConfig,
    ImageCalibrationConfig,
    LinkCalibrationConfig,
    RandomCalibrationConfig,
    RVC2Config,
    RVC3Config,
    RVC4Config,
    SingleStageConfig,
)
from modelconverter.utils.target_versions import (
    get_default_target_version,
)
from modelconverter.utils.types import Target

CONVERSION_RUN_ID_ENV_VAR = "MODELCONVERTER_CONVERSION_RUN_ID"
FLOW_NAME = "modelconverter_conversion_lifecycle"
COMMAND_EVENT = "modelconverter_command_ran"
CONFIGURED_EVENT = "modelconverter_conversion_configured"
RESULT_EVENT = "modelconverter_conversion_result_recorded"
MODELCONVERTER_TELEMETRY_DEFAULTS = TelemetryDefaults(
    enabled=True,
    backend="posthog",
    api_key="phc_ojEByaCiZZ5eigzaM43PaEVbfLfFDF5NgkXEMPabrT9a",
    endpoint="https://us.i.posthog.com",
)


def get_conversion_run_id() -> str:
    conversion_run_id = os.environ.get(CONVERSION_RUN_ID_ENV_VAR)
    if conversion_run_id:
        return conversion_run_id
    conversion_run_id = str(uuid4())
    os.environ[CONVERSION_RUN_ID_ENV_VAR] = conversion_run_id
    return conversion_run_id


def telemetry_environment() -> dict[str, str]:
    env = {}
    for key in [
        CONVERSION_RUN_ID_ENV_VAR,
        "LUXONIS_TELEMETRY_ENABLED",
        "LUXONIS_TELEMETRY_BACKEND",
        "LUXONIS_TELEMETRY_API_KEY",
        "LUXONIS_TELEMETRY_ENDPOINT",
        "LUXONIS_TELEMETRY_DEBUG",
        "LUXONIS_TELEMETRY_IS_LUXONIS_CLOUD",
    ]:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


def get_component_telemetry() -> Telemetry:
    return get_or_init(
        "modelconverter",
        library_version=__version__,
        config=TelemetryConfig.from_environ(
            defaults=MODELCONVERTER_TELEMETRY_DEFAULTS
        ),
        system_context_providers=[system_context_provider],
    )


def build_command_properties(
    *,
    conversion_run_id: str,
    target: Target,
    runs_in_docker: bool,
    dev_image: bool,
    gpu_enabled: bool,
    target_tool_version: str | None,
    custom_image_provided: bool,
    memory_limit_set: bool,
    cpu_limit_set: bool,
    result: str,
    duration_ms: int,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    return _drop_none(
        {
            "conversion_run_id": conversion_run_id,
            "command_name": "convert",
            "target": target.value,
            "runs_in_docker": runs_in_docker,
            "dev_image": dev_image,
            "gpu_enabled": gpu_enabled,
            "target_tool_version": target_tool_version,
            "custom_image_provided": custom_image_provided,
            "memory_limit_set": memory_limit_set,
            "cpu_limit_set": cpu_limit_set,
            "result": result,
            "failure_reason": failure_reason,
            "duration_ms": duration_ms,
        }
    )


def build_conversion_summary(
    cfg: Config,
    *,
    target: Target,
    config_source: str,
    archive_output_mode: str,
    archive_preprocess: bool,
    main_stage_provided: bool,
) -> dict[str, Any]:
    stages = list(cfg.stages.values())
    inputs = [inp for stage in stages for inp in stage.inputs]
    outputs = [out for stage in stages for out in stage.outputs]
    input_formats = [stage.input_file_type for stage in stages]
    input_format_values = [
        file_type.value.lower() for file_type in input_formats
    ]
    target_configs = [stage.get_target_config(target) for stage in stages]
    disable_calibration = any(
        target_cfg.disable_calibration for target_cfg in target_configs
    )
    calibration_sources = {
        _calibration_source(inp.calibration)
        for inp in inputs
        if _calibration_source(inp.calibration) is not None
    }

    return _drop_none(
        {
            "target": target.value,
            "config_source": config_source,
            "stage_count_bucket": bucket_count(len(stages)),
            "is_multistage": len(stages) > 1,
            "main_stage_provided": main_stage_provided,
            "archive_output_mode": archive_output_mode,
            "archive_preprocess": archive_preprocess,
            "input_model_format": (
                input_format_values[0] if input_format_values else None
            ),
            "multiple_input_model_formats": (
                len(set(input_formats)) > 1 if input_formats else None
            ),
            "input_count_bucket": bucket_count(len(inputs))
            if inputs
            else None,
            "output_count_bucket": (
                bucket_count(len(outputs)) if outputs else None
            ),
            "calibration_source": (
                next(iter(calibration_sources))
                if len(calibration_sources) == 1 and not disable_calibration
                else None
            ),
            "disable_calibration": disable_calibration,
            "keep_intermediate_outputs": any(
                stage.keep_intermediate_outputs for stage in stages
            ),
            "disable_onnx_simplification": all(
                stage.onnx_simplification is False for stage in stages
            ),
            "disable_onnx_optimization": all(
                stage.onnx_optimizations.all_disabled() for stage in stages
            ),
            "has_remote_output": any(
                stage.output_remote_url is not None for stage in stages
            ),
            "has_remote_intermediate_output": any(
                stage.intermediate_outputs_remote_url is not None
                for stage in stages
            ),
            "target_configuration": _target_configuration(target, stages),
        }
    )


def build_flow_properties(
    conversion_run_id: str, flow_step: str, properties: dict[str, Any]
) -> dict[str, Any]:
    return {
        "flow_name": FLOW_NAME,
        "conversion_run_id": conversion_run_id,
        "flow_step": flow_step,
        **properties,
    }


def build_conversion_result_properties(
    *,
    result: str,
    duration_ms: int,
    uploaded_output: bool,
    uploaded_intermediate_outputs: bool,
    failure_reason: str | None = None,
    output_artifact_count: int | None = None,
    peak_ram_bytes: int | None = None,
) -> dict[str, Any]:
    return _drop_none(
        {
            "result": result,
            "failure_reason": failure_reason,
            "duration_ms": duration_ms,
            "output_artifact_count_bucket": (
                bucket_count(output_artifact_count)
                if output_artifact_count is not None
                else None
            ),
            "uploaded_output": uploaded_output,
            "uploaded_intermediate_outputs": uploaded_intermediate_outputs,
            "peak_ram_usage_bucket": (
                bucket_memory_bytes(peak_ram_bytes)
                if peak_ram_bytes is not None
                else None
            ),
        }
    )


def command_result_from_exception(exc: BaseException | None) -> str:
    if exc is None:
        return "success"
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) and getattr(
        exc, "code", None
    ) in {None, 130}:
        return "interrupted"
    return "failed"


def command_failure_reason_from_exception(
    exc: BaseException | None,
) -> str | None:
    if exc is None:
        return None
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) and getattr(
        exc, "code", None
    ) in {None, 130}:
        return "user_interrupt"
    return "runtime_error"


def runtime_failure_reason_from_exception(
    exc: BaseException | None, *, phase: str
) -> str | None:
    if exc is None:
        return None
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) and getattr(
        exc, "code", None
    ) in {None, 130}:
        return "user_interrupt"
    if phase == "configuration":
        return "config_error"
    if phase.startswith("upload"):
        return "upload_error"
    return "conversion_error"


def resolve_target_tool_version(
    *, target: Target, tool_version: str | None, image: str | None
) -> str | None:
    if image is not None and ":" in image.rsplit("/", maxsplit=1)[-1]:
        return None
    return tool_version or get_default_target_version(target.value)


def detect_config_source(
    path: str | None,
    opts: list[str],
    archive_cfg: NNArchiveConfig | None,
) -> str:
    if archive_cfg is not None:
        if path and is_nn_archive(path):
            return "nn_archive"
        return "archive_directory"
    if path and _looks_like_model_input(path):
        return "direct_model_input"
    if "input_model" in opts[::2]:
        return "direct_model_input"
    return "yaml_config"


def _looks_like_model_input(path: str) -> bool:
    suffixes = {suffix.lower() for suffix in Path(path).suffixes}
    return bool(
        suffixes & {".onnx", ".xml", ".bin", ".dlc", ".tflite", ".pt", ".pth"}
    )


def bucket_count(value: int | None) -> str | None:
    if value is None:
        return None
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2_4"
    return "5_plus"


def bucket_memory_bytes(value: int | None) -> str | None:
    if value is None:
        return None
    mib = value / (1024 * 1024)
    if mib < 512:
        return "under_512m"
    if mib < 1024:
        return "512m_1g"
    if mib < 4096:
        return "1g_4g"
    return "above_4g"


def peak_ram_usage_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak)
    return int(peak * 1024)


def _target_configuration(
    target: Target, stages: list[SingleStageConfig]
) -> dict[str, Any] | None:
    first_stage = stages[0]
    target_config = first_stage.get_target_config(target)

    if isinstance(target_config, RVC2Config):
        return _drop_none(
            {
                "number_of_shaves": target_config.number_of_shaves,
                "superblob": target_config.superblob,
                "n_workers_bucket": bucket_count(target_config.n_workers),
                "compress_to_fp16": target_config.compress_to_fp16,
            }
        )

    if isinstance(target_config, RVC3Config):
        return {
            "pot_target_device": target_config.pot_target_device.value.lower(),
            "compress_to_fp16": target_config.compress_to_fp16,
        }

    if isinstance(target_config, RVC4Config):
        return _drop_none(
            {
                "quantization_mode": (
                    target_config.quantization_mode.value.lower()
                ),
                "optimization_level": target_config.optimization_level,
                "use_per_channel_quantization": (
                    target_config.use_per_channel_quantization
                ),
                "use_per_row_quantization": (
                    target_config.use_per_row_quantization
                ),
                "keep_raw_images": target_config.keep_raw_images,
                "htp_soc_count_bucket": bucket_count(
                    len(target_config.htp_socs)
                ),
                "has_quantization_overrides": (
                    target_config.encodings is not None
                ),
            }
        )

    if isinstance(target_config, HailoConfig):
        return _drop_none(
            {
                "optimization_level": target_config.optimization_level,
                "compression_level": target_config.compression_level,
                "batch_size_bucket": bucket_count(target_config.batch_size),
                "disable_compilation": target_config.disable_compilation,
                "hw_arch": target_config.hw_arch,
                "alls_count_bucket": bucket_count(len(target_config.alls)),
            }
        )

    return None


def _calibration_source(calibration: Any) -> str | None:
    if isinstance(calibration, ImageCalibrationConfig):
        return "image_directory"
    if isinstance(calibration, RandomCalibrationConfig):
        return "random"
    if isinstance(calibration, LinkCalibrationConfig):
        return "remote_link"
    return None


def _drop_none(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in properties.items() if value is not None
    }
