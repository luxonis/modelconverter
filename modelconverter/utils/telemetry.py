"""Telemetry events reported by the modelconverter CLI.

Builds the property payloads that the CLI sends through
``luxonis_ml.telemetry`` for a conversion run: which command ran, how
the conversion was configured, and how it ended. The reported values are
enum values, booleans, numeric settings and bucketed counts derived from
the configuration rather than raw model names, paths or file contents. A
conversion run id, propagated through an environment variable, ties the
host-side command event to the events emitted from inside the conversion
Docker container.
"""

import os
import resource
import sys
from enum import Enum
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


class TelemetryFlowStep(str, Enum):
    """Step of the conversion lifecycle a flow event belongs to."""

    CONFIGURATION_RESOLVED = "configuration_resolved"
    RESULT_RECORDED = "result_recorded"


class CommandName(str, Enum):
    """CLI command a command telemetry event was emitted for."""

    CONVERT = "convert"


class CommandResult(str, Enum):
    """Outcome of a command or of a conversion run."""

    SUCCESS = "success"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class FailureReason(str, Enum):
    """Reason a command or a conversion run did not succeed."""

    USER_INTERRUPT = "user_interrupt"
    RUNTIME_ERROR = "runtime_error"
    CONFIG_ERROR = "config_error"
    UPLOAD_ERROR = "upload_error"
    CONVERSION_ERROR = "conversion_error"


class ConversionPhase(str, Enum):
    """Phase of the conversion that was active when it ended."""

    CONFIGURATION = "configuration"
    CONVERSION = "conversion"
    UPLOAD_OUTPUT = "upload_output"
    UPLOAD_INTERMEDIATE = "upload_intermediate"


class ConfigSource(str, Enum):
    """Origin of the configuration used for a conversion."""

    DIRECT_MODEL_INPUT = "direct_model_input"
    YAML_CONFIG = "yaml_config"
    NN_ARCHIVE = "nn_archive"
    ARCHIVE_DIRECTORY = "archive_directory"


class ArchiveOutputMode(str, Enum):
    """Form in which the converted model is delivered.

    Either a plain model file (``native``) or a Luxonis NN Archive
    (``nn_archive``).
    """

    NATIVE = "native"
    NN_ARCHIVE = "nn_archive"


class CalibrationSource(str, Enum):
    """Kind of calibration data configured for an input."""

    IMAGE_DIRECTORY = "image_directory"
    RANDOM = "random"
    REMOTE_LINK = "remote_link"


def get_conversion_run_id() -> str:
    """Return the id of the current conversion run.

    The id is stored in the ``MODELCONVERTER_CONVERSION_RUN_ID``
    environment variable and generated on first use, so that the
    host-side command and the conversion running inside the Docker
    container report the same id.

    Returns:
        Conversion run id for the current process.

    """
    conversion_run_id = os.environ.get(CONVERSION_RUN_ID_ENV_VAR)
    if conversion_run_id:
        return conversion_run_id
    conversion_run_id = str(uuid4())
    os.environ[CONVERSION_RUN_ID_ENV_VAR] = conversion_run_id
    return conversion_run_id


def telemetry_environment() -> dict[str, str]:
    """Collect the telemetry variables to forward to the container.

    Returns:
        The conversion run id and ``LUXONIS_TELEMETRY_*`` variables that
        are set in the current environment. Unset variables are omitted.

    """
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
    """Return the shared modelconverter telemetry client.

    The client is created on first use, configured from the environment
    with the modelconverter telemetry defaults as fallback.

    Returns:
        The ``Telemetry`` instance registered for modelconverter.

    """
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
    result: CommandResult,
    duration_ms: int,
    failure_reason: FailureReason | None = None,
) -> dict[str, Any]:
    """Build the properties of the command telemetry event.

    Args:
        conversion_run_id: Id tying the event to the conversion run.
        target: Conversion target the command was invoked for.
        runs_in_docker: Whether the conversion runs inside a Docker
            container.
        dev_image: Whether the development image tag is used.
        gpu_enabled: Whether GPU access was requested.
        target_tool_version: Resolved tool version of the target, or
            ``None`` if it is determined by a custom image tag.
        custom_image_provided: Whether a custom Docker image was
            given. The image name itself is never reported.
        memory_limit_set: Whether a memory limit was given. The limit
            itself is never reported.
        cpu_limit_set: Whether a CPU limit was given. The limit itself
            is never reported.
        result: Outcome of the command.
        duration_ms: Wall-clock duration of the command in
            milliseconds.
        failure_reason: Reason the command failed, or ``None`` if it
            did not fail.

    Returns:
        Event properties, with the ``None`` values dropped.

    """
    return _drop_none(
        {
            "conversion_run_id": conversion_run_id,
            "command_name": CommandName.CONVERT.value,
            "target": target.value,
            "runs_in_docker": runs_in_docker,
            "dev_image": dev_image,
            "gpu_enabled": gpu_enabled,
            "target_tool_version": target_tool_version,
            "custom_image_provided": custom_image_provided,
            "memory_limit_set": memory_limit_set,
            "cpu_limit_set": cpu_limit_set,
            "result": result.value,
            "failure_reason": failure_reason.value if failure_reason else None,
            "duration_ms": duration_ms,
        }
    )


def build_conversion_summary(
    cfg: Config,
    *,
    target: Target,
    config_source: ConfigSource,
    archive_output_mode: ArchiveOutputMode,
    archive_preprocess: bool,
    main_stage_provided: bool,
) -> dict[str, Any]:
    """Summarize a resolved conversion configuration for telemetry.

    Only enum values, booleans, numeric settings and bucketed counts
    derived from the configuration are reported.

    Args:
        cfg: Resolved conversion configuration.
        target: Conversion target.
        config_source: Where the configuration came from.
        archive_output_mode: Form in which the outputs are delivered.
        archive_preprocess: Whether the pre-processing is added to the
            NN archive instead of the model.
        main_stage_provided: Whether the name of the main stage was
            given explicitly.

    Returns:
        Event properties, with the ``None`` values dropped.

    """
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
            "config_source": config_source.value,
            "stage_count_bucket": bucket_count(len(stages)),
            "is_multistage": len(stages) > 1,
            "main_stage_provided": main_stage_provided,
            "archive_output_mode": archive_output_mode.value,
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
    conversion_run_id: str,
    flow_step: TelemetryFlowStep,
    properties: dict[str, Any],
) -> dict[str, Any]:
    """Extend event properties with the conversion flow metadata.

    Args:
        conversion_run_id: Id tying the event to the conversion run.
        flow_step: Step of the conversion flow the event belongs to.
        properties: Event properties to extend.

    Returns:
        The properties, with the flow name, the conversion run id and
        the flow step added.

    """
    return {
        "flow_name": FLOW_NAME,
        "conversion_run_id": conversion_run_id,
        "flow_step": flow_step.value,
        **properties,
    }


def build_conversion_result_properties(
    *,
    result: CommandResult,
    duration_ms: int,
    uploaded_output: bool,
    uploaded_intermediate_outputs: bool,
    failure_reason: FailureReason | None = None,
    output_artifact_count: int | None = None,
    peak_ram_bytes: int | None = None,
) -> dict[str, Any]:
    """Build the properties of the conversion result event.

    Args:
        result: Outcome of the conversion.
        duration_ms: Wall-clock duration of the conversion in
            milliseconds.
        uploaded_output: Whether the output models were uploaded to a
            remote location.
        uploaded_intermediate_outputs: Whether the intermediate outputs
            were uploaded to a remote location.
        failure_reason: Reason the conversion failed, or ``None`` if it
            did not fail.
        output_artifact_count: Number of produced output artifacts.
            Bucketed by `bucket_count` before it is reported.
        peak_ram_bytes: Peak RAM usage in bytes. Bucketed by
            `bucket_memory_bytes` before it is reported.

    Returns:
        Event properties, with the ``None`` values dropped.

    """
    return _drop_none(
        {
            "result": result.value,
            "failure_reason": failure_reason.value if failure_reason else None,
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


def command_result_from_exception(
    exc: BaseException | None,
) -> CommandResult:
    """Derive the command result from the exception that ended it.

    Args:
        exc: Exception that terminated the command, or ``None`` if the
            command completed normally.

    Returns:
        ``CommandResult.SUCCESS`` for no exception or a ``SystemExit``
        with code ``None`` or ``0``, ``CommandResult.INTERRUPTED`` for a
        keyboard interrupt or an exit code of ``130``, and
        ``CommandResult.FAILED`` otherwise.

    """
    if exc is None:
        return CommandResult.SUCCESS
    code = getattr(exc, "code", None)
    if isinstance(exc, SystemExit) and code in {None, 0}:
        return CommandResult.SUCCESS
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) and code in {
        None,
        130,
    }:
        return CommandResult.INTERRUPTED
    return CommandResult.FAILED


def command_failure_reason_from_exception(
    exc: BaseException | None,
) -> FailureReason | None:
    """Derive the command failure reason from the exception.

    Args:
        exc: Exception that terminated the command, or ``None`` if the
            command completed normally.

    Returns:
        ``None`` if the command did not fail,
        ``FailureReason.USER_INTERRUPT`` for a keyboard interrupt or an
        exit code of ``130``, and ``FailureReason.RUNTIME_ERROR``
        otherwise.

    """
    if exc is None:
        return None
    code = getattr(exc, "code", None)
    if isinstance(exc, SystemExit) and code in {None, 0}:
        return None
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) and code in {
        None,
        130,
    }:
        return FailureReason.USER_INTERRUPT
    return FailureReason.RUNTIME_ERROR


def runtime_failure_reason_from_exception(
    exc: BaseException | None, *, phase: ConversionPhase
) -> FailureReason | None:
    """Derive the conversion failure reason from the exception.

    Args:
        exc: Exception that ended the conversion, or ``None`` if the
            conversion finished normally.
        phase: Conversion phase that was active when it ended.

    Returns:
        ``None`` if there was no exception,
        ``FailureReason.USER_INTERRUPT`` for a keyboard interrupt or an
        exit code of ``130``, and otherwise the failure reason matching
        the phase.

    """
    if exc is None:
        return None
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) and getattr(
        exc, "code", None
    ) in {None, 130}:
        return FailureReason.USER_INTERRUPT
    if phase is ConversionPhase.CONFIGURATION:
        return FailureReason.CONFIG_ERROR
    if phase in {
        ConversionPhase.UPLOAD_OUTPUT,
        ConversionPhase.UPLOAD_INTERMEDIATE,
    }:
        return FailureReason.UPLOAD_ERROR
    return FailureReason.CONVERSION_ERROR


def resolve_target_tool_version(
    *, target: Target, tool_version: str | None, image: str | None
) -> str | None:
    """Resolve the tool version the target is converted with.

    Args:
        target: Conversion target.
        tool_version: Explicitly requested tool version, if any.
        image: Custom Docker image, if any.

    Returns:
        The requested tool version, or the default one for the target.
        ``None`` is returned when a custom image with an explicit tag is
        given, since the version is then decided by that tag.

    """
    if image is not None and ":" in image.rsplit("/", maxsplit=1)[-1]:
        return None
    return tool_version or get_default_target_version(target.value)


def detect_config_source(
    path: str | None,
    opts: list[str],
    archive_cfg: NNArchiveConfig | None,
) -> ConfigSource:
    """Detect where the conversion configuration came from.

    Args:
        path: Path given on the command line, if any.
        opts: CLI configuration overrides as alternating key and value
            tokens.
        archive_cfg: Configuration extracted from an NN archive, if the
            input was an archive or an archive directory.

    Returns:
        ``ConfigSource.NN_ARCHIVE`` or
        ``ConfigSource.ARCHIVE_DIRECTORY`` when an archive
        configuration is given, ``ConfigSource.DIRECT_MODEL_INPUT``
        when a model file or an ``input_model`` override is given, and
        ``ConfigSource.YAML_CONFIG`` otherwise.

    """
    if archive_cfg is not None:
        if path and is_nn_archive(path):
            return ConfigSource.NN_ARCHIVE
        return ConfigSource.ARCHIVE_DIRECTORY
    if path and _looks_like_model_input(path):
        return ConfigSource.DIRECT_MODEL_INPUT
    if "input_model" in opts[::2]:
        return ConfigSource.DIRECT_MODEL_INPUT
    return ConfigSource.YAML_CONFIG


def _looks_like_model_input(path: str) -> bool:
    suffixes = {suffix.lower() for suffix in Path(path).suffixes}
    return bool(
        suffixes & {".onnx", ".xml", ".bin", ".dlc", ".tflite", ".pt", ".pth"}
    )


def bucket_count(value: int | None) -> str | None:
    """Reduce a count to a coarse bucket label.

    Args:
        value: Count to report. Bucketing keeps a telemetry event from
            carrying the exact shape of the converted model.

    Returns:
        One of ``"0"``, ``"1"``, ``"2_4"`` or ``"5_plus"``, or ``None``
        if ``value`` is ``None``.

    """
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
    """Reduce a memory size in bytes to a coarse bucket label.

    Args:
        value: Memory size in bytes, as `peak_ram_usage_bytes` measures
            it.

    Returns:
        One of ``"under_512m"``, ``"512m_1g"``, ``"1g_4g"`` or
        ``"above_4g"``, or ``None`` if ``value`` is ``None``.

    """
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
    """Return the peak RAM usage of this process in bytes.

    The value comes from ``resource.getrusage``, whose ``ru_maxrss`` is
    reported in bytes on macOS and in kibibytes elsewhere.

    Returns:
        Peak resident set size in bytes.

    """
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
        return _drop_none(
            {
                "pot_target_device": (
                    target_config.pot_target_device.value.lower()
                ),
                "compress_to_fp16": target_config.compress_to_fp16,
            }
        )

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


def _calibration_source(calibration: Any) -> CalibrationSource | None:
    if isinstance(calibration, ImageCalibrationConfig):
        return CalibrationSource.IMAGE_DIRECTORY
    if isinstance(calibration, RandomCalibrationConfig):
        return CalibrationSource.RANDOM
    if isinstance(calibration, LinkCalibrationConfig):
        return CalibrationSource.REMOTE_LINK
    return None


def _drop_none(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in properties.items() if value is not None
    }
