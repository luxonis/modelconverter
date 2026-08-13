"""Entry point of the ``modelconverter`` command-line interface.

Defines the commands the CLI exposes: converting a model for a target
platform, running inference, benchmarking and analyzing it on a device,
visualizing the analysis, packaging a model into an NN Archive, opening
a shell in a target's container and managing the cache. The commands
that need the vendor conversion tools -- ``convert``, ``infer`` and
``shell`` -- are re-run inside the Docker image of the requested target
platform unless they already run inside one; the others run on the
host.
"""

import importlib.metadata
import os
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from cyclopts import App, Group, Parameter
from loguru import logger
from luxonis_ml.nn_archive import ArchiveGenerator, is_nn_archive
from luxonis_ml.utils import LuxonisFileSystem, setup_logging
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from modelconverter.cli import (
    extract_preprocessing,
    get_configs,
    get_output_dir_name,
    init_dirs,
    resolve_output_dir,
)
from modelconverter.packages import (
    get_analyzer,
    get_benchmark,
    get_exporter,
    get_inferer,
    get_visualizer,
)
from modelconverter.packages.multistage_exporter import MultiStageExporter
from modelconverter.utils import (
    ModelconverterException,
    archive_from_model,
    docker_build,
    docker_exec,
    get_default_target_version,
    get_local_docker_image,
    in_docker,
    resolve_path,
    upload_to_remote,
)
from modelconverter.utils.config import SingleStageConfig
from modelconverter.utils.constants import (
    CONVERSION_MARKER,
    MODELS_DIR,
    get_cache_dir,
)
from modelconverter.utils.general import (
    dir_stats,
    human_size,
    parse_size,
    sanitize_net_name,
)
from modelconverter.utils.input_staging import (
    cache_budget,
    cache_is_in_use,
    path_flags_for,
    stage_inputs,
)
from modelconverter.utils.nn_archive import generate_archive
from modelconverter.utils.telemetry import (
    COMMAND_EVENT,
    CONFIGURED_EVENT,
    CONVERSION_RUN_ID_ENV_VAR,
    RESULT_EVENT,
    ArchiveOutputMode,
    CommandResult,
    ConversionPhase,
    FailureReason,
    TelemetryFlowStep,
    build_command_properties,
    build_conversion_result_properties,
    build_conversion_summary,
    build_flow_properties,
    command_failure_reason_from_exception,
    command_result_from_exception,
    detect_config_source,
    get_component_telemetry,
    get_conversion_run_id,
    peak_ram_usage_bytes,
    resolve_target_tool_version,
    runtime_failure_reason_from_exception,
)
from modelconverter.utils.types import Target

app = App(
    name="Modelconverter",
    version=lambda: (
        f"ModelConverter v{importlib.metadata.version('modelconv')}"
    ),
)

app.meta.group_parameters = Group("Global Parameters", sort_key=0)
app["--help"].group = app.meta.group_parameters
app["--version"].group = app.meta.group_parameters

docker_parameters = Group.create_ordered(
    "Docker Parameters", help="Global parameters for all docker commands"
)
docker_commands = Group.create_ordered("Docker Commands")
device_commands = Group.create_ordered("Device Commands")


@contextmanager
def catch_exceptions():
    """Log any exception raised in the block and exit.

    Exits with status 1 for a ``ModelconverterException`` and with
    status 2 for any other exception.
    """
    try:
        yield
    except ModelconverterException:
        logger.exception("Encountered an exception in the conversion process!")
        sys.exit(1)
    except Exception:
        logger.exception("Encountered an unexpected error!")
        sys.exit(2)


@app.command(group=docker_commands)
def convert(
    target: Target,
    /,
    *opts: str,
    path: str | None = None,
    output_dir: str | None = None,
    to: Literal["native", "nn_archive"] = "nn_archive",
    main_stage: str | None = None,
    archive_preprocess: bool = False,
) -> None:
    """Export the model for the specified target platform.

    Args:
        target: The target platform to export the model for.
        opts: A list of optional CLI overrides for the configuration
            file.
        path: A URL or a path to the configuration file, NN Archive
            or a standalone model file.
        output_dir: Name of the directory where the exported model will
            be saved.
        to: Whether to export the model to a simple model file or a
            Luxonis NN Archive.
        main_stage: Name of the stage with the main model.
            Only needed for multistage configs and when converting to
            NN Archive. When converting from NN Archive, the stage names
            are named the same as the model files without the suffix.
        archive_preprocess: Add the pre-processing to the NN archive
            instead of the model. In case of conversion from archive to
            archive, it moves the preprocessing to the new archive.

    """

    def handle_signal(signum: int, frame: Any) -> None:
        signame = signal.Signals(signum).name
        logger.error(f"{signame} received, exiting...")
        sys.exit(130)

    signal.signal(signal.SIGTERM, handle_signal)

    if output_dir is not None:
        output_dir = sanitize_net_name(output_dir)
    t = time.monotonic()
    runtime_telemetry = get_component_telemetry()
    conversion_run_id = get_conversion_run_id()
    original_path = path
    opts: list[str] = list(opts or [])
    conversion_start: float | None = None
    conversion_summary: dict[str, Any] | None = None
    output_artifact_count: int | None = None
    uploaded_output = False
    uploaded_intermediate_outputs = False
    phase = ConversionPhase.CONFIGURATION
    caught_exc: BaseException | None = None

    try:
        main_stage_provided = main_stage is not None
        if path is not None:
            suffix = Path(path).suffix
            if suffix in {".xml", ".bin"} and target not in {
                Target.RVC2,
                Target.RVC3,
            }:
                raise ValueError(
                    f"OpenVINO IR format is not supported for target {target.name}."
                )
            if suffix in {".onnx", ".xml", ".dlc", ".tflite"}:
                opts = ["input_model", path, *opts]
                if suffix == ".xml":
                    opts = [
                        "input_bin",
                        str(Path(path).with_suffix(".bin")),
                        *opts,
                    ]
                path = None
            elif suffix == ".bin":
                opts = [
                    "input_model",
                    str(Path(path).with_suffix(".xml")),
                    "input_bin",
                    path,
                    *opts,
                ]
                path = None

        init_dirs()
        cfg, archive_cfg, _main_stage = get_configs(target, path, list(opts))
        main_stage = main_stage or _main_stage
        is_multistage = len(cfg.stages) > 1
        if is_multistage and main_stage is None:
            raise ValueError(
                "Main stage name must be provided for multistage models."
            )
        preprocessing = {}
        if archive_preprocess:
            cfg, preprocessing = extract_preprocessing(cfg)

        output_path = get_output_dir_name(target, cfg.name, output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / CONVERSION_MARKER).touch()
        setup_logging(
            file=str(output_path / "modelconverter.log"),
            use_rich=cfg.rich_logging,
        )
        if is_multistage:
            exporter = MultiStageExporter(
                target=target, config=cfg, output_dir=output_path
            )
        else:
            exporter = get_exporter(
                target,
                config=next(iter(cfg.stages.values())),
                output_dir=output_path,
            )

        conversion_summary = build_flow_properties(
            conversion_run_id,
            TelemetryFlowStep.CONFIGURATION_RESOLVED,
            build_conversion_summary(
                cfg,
                target=target,
                config_source=detect_config_source(
                    original_path, opts, archive_cfg
                ),
                archive_output_mode=ArchiveOutputMode(to),
                archive_preprocess=archive_preprocess,
                main_stage_provided=main_stage_provided,
            ),
        )
        runtime_telemetry.capture(
            CONFIGURED_EVENT,
            conversion_summary,
            include_system_metadata=True,
            distinct_id=conversion_run_id,
        )

        conversion_start = time.monotonic()
        phase = ConversionPhase.CONVERSION
        out_models = exporter.run()
        if not isinstance(out_models, list):
            out_models = [out_models]
        if to == "nn_archive":
            from modelconverter.packages.base_exporter import Exporter

            archive_name = None
            if original_path is not None and is_nn_archive(original_path):
                archive_filename = Path(original_path).name
                archive_suffix = "".join(Path(original_path).suffixes)
                archive_name = (
                    archive_filename.removesuffix(archive_suffix)
                    if archive_suffix
                    else archive_filename
                )

            assert main_stage is not None
            out_models = [
                generate_archive(
                    target=target,
                    cfg=cfg,
                    main_stage=main_stage,
                    out_models=out_models,
                    output_path=output_path,
                    archive_cfg=archive_cfg,
                    preprocessing=preprocessing,
                    inference_model_path=(
                        exporter.inference_model_path
                        if isinstance(exporter, Exporter)
                        else exporter.exporters[
                            main_stage
                        ].inference_model_path
                    ),
                    archive_name=archive_name,
                )
            ]
        output_artifact_count = len(out_models)

        if isinstance(exporter.config, SingleStageConfig):
            _cfg = exporter.config
        else:
            _cfg = next(iter(exporter.config.stages.values()))
        upload_url = _cfg.output_remote_url
        intermediate_url = _cfg.intermediate_outputs_remote_url
        put_file_plugin = _cfg.put_file_plugin

        if upload_url is not None:
            phase = ConversionPhase.UPLOAD_OUTPUT
            for model_path in out_models:
                logger.info(f"Uploading {model_path} to {upload_url}")
                upload_to_remote(
                    model_path,
                    upload_url,
                    put_file_plugin,
                )
            uploaded_output = True

        if intermediate_url is not None:
            phase = ConversionPhase.UPLOAD_INTERMEDIATE
            exporters = (
                exporter.exporters.values()
                if isinstance(exporter, MultiStageExporter)
                else [exporter]
            )
            for exporter in exporters:
                logger.info(
                    f"Uploading intermediate outputs to {intermediate_url}"
                )
                upload_to_remote(
                    exporter.intermediate_outputs_dir,
                    intermediate_url,
                    put_file_plugin,
                )
            uploaded_intermediate_outputs = True

        logger.info("Conversion finished successfully")
    except KeyboardInterrupt as exc:
        caught_exc = exc
        logger.error("Keyboard interrupt received, exiting...")
        raise SystemExit(130) from exc
    except SystemExit as exc:
        caught_exc = exc
        raise
    except ModelconverterException as exc:
        caught_exc = exc
        logger.exception("Encountered an exception in the conversion process!")
        raise SystemExit(1) from exc
    except Exception as exc:
        caught_exc = exc
        logger.exception("Encountered an unexpected error!")
        raise SystemExit(2) from exc
    finally:
        peak_ram_bytes = peak_ram_usage_bytes()
        logger.info(f"Peak RAM usage: {peak_ram_bytes / (1024 * 1024):.2f} MB")
        logger.info(
            f"Conversion finished in {time.monotonic() - t:.2f} seconds"
        )
        failure_reason = runtime_failure_reason_from_exception(
            caught_exc, phase=phase
        )
        runtime_telemetry.capture(
            RESULT_EVENT,
            build_flow_properties(
                conversion_run_id,
                TelemetryFlowStep.RESULT_RECORDED,
                {
                    **(
                        {
                            key: value
                            for key, value in conversion_summary.items()
                            if key != "flow_step"
                        }
                        if conversion_summary is not None
                        else {"target": target.value}
                    ),
                    **build_conversion_result_properties(
                        result=(
                            CommandResult.SUCCESS
                            if caught_exc is None
                            else (
                                CommandResult.INTERRUPTED
                                if failure_reason
                                is FailureReason.USER_INTERRUPT
                                else CommandResult.FAILED
                            )
                        ),
                        failure_reason=failure_reason,
                        duration_ms=int(
                            (time.monotonic() - (conversion_start or t)) * 1000
                        ),
                        output_artifact_count=output_artifact_count,
                        uploaded_output=uploaded_output,
                        uploaded_intermediate_outputs=uploaded_intermediate_outputs,
                        peak_ram_bytes=peak_ram_bytes,
                    ),
                },
            ),
            include_system_metadata=True,
            distinct_id=conversion_run_id,
        )
        os.environ.pop(CONVERSION_RUN_ID_ENV_VAR, None)


@app.command(group=docker_commands)
def infer(
    target: Target,
    /,
    *opts: str,
    model_path: str,
    input_path: Path,
    output_dir: str,
    config: str | None = None,
    path: str | None = None,
    stage: str | None = None,
) -> None:
    """Run inference on the specified target platform.

    Args:
        target: The target platform to run the inference on.
        opts: A list of optional CLI overrides for the configuration
            file.
        model_path: A URL or a path to the model file.
        input_path: Path to the directory with data for inference.
            The directory must contain one subdirectory per input, named
            the same as the input. The files may be images, ``.npy``
            arrays or ``.raw`` buffers.
        output_dir: Name of the directory where the inference results
            will be saved.
        config: A URL or a path to the configuration file.
        path: An alias for ``config``. Deprecated.
        stage: Name of the stage to run. Only needed for multistage
            configs. If not provided, the first stage will be used.

    """
    if path is not None:
        config = path
    with catch_exceptions():
        mult_cfg, _, _ = get_configs(target, str(config), list(opts))
        cfg = mult_cfg.get_stage_config(stage)
        output_path = resolve_output_dir(output_dir)
        setup_logging(
            file=str(output_path.parent / f"{output_path.name}.log"),
            use_rich=mult_cfg.rich_logging,
        )
        logger.info("Starting inference")
        get_inferer(target, model_path, input_path, output_path, cfg).run()


@app.command(group=docker_commands)
def shell(
    target: Target,
    /,
    *,
    command: Annotated[str | None, Parameter(name=["-c", "--command"])] = None,
) -> None:
    """Boot up a shell inside a docker container for the specified
    target platform.

    Args:
        target: The target platform.
        command: The command to run in the shell. If not provided, a
            bash shell is started. If you want to run a command with
            arguments, use quotes around the command.

    """
    args = ["bash"]
    if command is not None:
        args.extend(["-c", command])
    os.execle("/bin/bash", *args, os.environ)


@app.meta.command(group=device_commands)
def benchmark(
    target: Target,
    /,
    *,
    model_path: str,
    full: bool = False,
    save: bool = False,
    repetitions: Annotated[int, Parameter(group=["RVC2", "RVC4"])] = 10,
    benchmark_time: Annotated[int, Parameter(group=["RVC2", "RVC4"])] = 20,
    num_threads: Annotated[int, Parameter(group=["RVC2", "RVC4"])] = 2,
    num_messages: Annotated[int, Parameter(group=["RVC2", "RVC4"])] = 50,
    requests: Annotated[int, Parameter(group="RVC3")] = 1,
    profile: Annotated[
        Literal[
            "low_balanced",
            "balanced",
            "high_performance",
            "sustained_high_performance",
            "burst",
            "low_power_saver",
            "power_saver",
            "high_power_saver",
            "extreme_power_saver",
            "system_settings",
        ],
        Parameter(group="RVC4"),
    ] = "balanced",
    runtime: Annotated[Literal["dsp", "cpu"], Parameter(group="RVC4")] = "dsp",
    num_images: Annotated[int, Parameter(group="RVC4")] = 500,
    device_ip: Annotated[str | None, Parameter(group="RVC4")] = None,
    device_id: Annotated[str | None, Parameter(group="RVC4")] = None,
    dai_benchmark: Annotated[bool, Parameter(group="RVC4")] = True,
    device_monitor: Annotated[bool, Parameter(group="RVC4")] = True,
) -> None:
    """Run benchmark on the specified target platform.

    Args:
        target: The target platform to run the benchmark on.
        model_path: A URL or a path to the model file.
        full: If ``True``, runs the full benchmark using all
            configurations.
        save: If ``True``, saves the benchmark results to a file.
        repetitions: The number of repetitions to perform. Only relevant
            for DAI benchmark.
        benchmark_time: The duration in seconds for time-based
            benchmarking (overrides repetitions).
        num_threads: The number of threads to use for inference. Only
            relevant for DAI benchmark.
        num_messages: The number of messages to measure for each report.
            Only relevant for DAI benchmark.
        requests: The number of requests to perform.
        profile: The SNPE profile to use for inference.
        runtime: The SNPE runtime to use for inference (dsp or cpu).
        num_images: The number of images to use for inference. Only
            relevant for SNPE backend.
        device_ip: IP address of the device to run the benchmark on.
            Interchangeable with ``device_id``. If neither is given, DAI
            selects the default device. If both are given, ``device_id``
            takes precedence.
        device_id: The unique ID of the device to run the benchmark on.
            Interchangeable with ``device_ip``. If neither is given, DAI
            selects the default device. If both are given, ``device_id``
            takes precedence.
        dai_benchmark: Whether to run the benchmark using the DAI V3. If
            ``False``, the SNPE tools are used.
        device_monitor: Whether to monitor the device performance during
            benchmarking and include it in the results. Only relevant
            for RVC4 target.

    """
    if target in {Target.RVC2, Target.RVC4}:
        kwargs = {
            "repetitions": repetitions,
            "benchmark_time": benchmark_time,
            "num_threads": num_threads,
            "num_messages": num_messages,
        }
        if target is Target.RVC4:
            kwargs |= {
                "profile": profile,
                "runtime": runtime,
                "num_images": num_images,
                "dai_benchmark": dai_benchmark,
                "device_ip": device_ip,
                "device_id": device_id,
                "device_monitor": device_monitor,
            }
    elif target is Target.RVC3:
        kwargs = {
            "requests": requests,
        }
    get_benchmark(target, model_path).run(full=full, save=save, **kwargs)


@app.meta.command(group=device_commands)
def analyze(
    *,
    device_ip: str | None = None,
    device_id: str | None = None,
    dlc_model_path: str,
    onnx_model_path: str,
    image_subset: int | None = None,
    image_dirs: Annotated[
        list[str], Parameter(negative_iterable=[], consume_multiple=True)
    ],
    analyze_outputs: bool = True,
    analyze_cycles: bool = True,
) -> None:
    """Run layer and cycle analysis on the specified DLC model.

    Requires the RVC4 device to be connected and accessible using
    the ``adb`` command.

    Args:
        device_ip: IP address of the device to run the benchmark on.
            Interchangeable with ``device_id``. If neither is given, DAI
            selects the default device. If both are given, ``device_id``
            takes precedence.
        device_id: DeviceId of the device to run the benchmark on.
            Interchangeable with ``device_ip``. If neither is given, DAI
            selects the default device. If both are given, ``device_id``
            takes precedence.
        dlc_model_path: The path to the DLC model file.
        onnx_model_path: The path to the corresponding ONNX model file
            that was used for converting to DLC.
        image_subset: If provided, limit analysis to the first N
            supported input files per input directory.
        image_dirs: A list of names and paths to directories with images
            for each input of the model.
        analyze_outputs: Whether to analyze the layer outputs.
        analyze_cycles: Whether to analyze the layer cycles.

    """
    with catch_exceptions():
        logger.info("Starting analysis")
        if image_subset is not None and image_subset <= 0:
            raise ValueError("image_subset must be a positive integer.")
        if len(image_dirs) == 1:
            image_dirs_dict = {"default": image_dirs[0]}
        else:
            if len(image_dirs) % 2 != 0:
                raise ValueError(
                    "Please supply the same amount of model input names and test image directories."
                )
            image_dirs_dict = {
                image_dirs[i]: image_dirs[i + 1]
                for i in range(0, len(image_dirs), 2)
            }

        analyzer = get_analyzer(
            Target.RVC4,
            device_ip,
            device_id,
            dlc_model_path,
            image_dirs_dict,
            image_subset=image_subset,
        )
        if analyze_outputs:
            logger.info("Analyzing layer outputs")
            analyzer.analyze_layer_outputs(
                resolve_path(onnx_model_path, Path.cwd())
            )
        if analyze_cycles:
            logger.info("Analyzing layer cycles")
            analyzer.analyze_layer_cycles()
        logger.info("Analysis finished successfully")


@app.meta.command
def visualize(dir_path: str) -> None:
    """Visualize the analysis results.

    Args:
        dir_path: The path to the directory containing the analysis
            results. The default search path is ``output/analysis``.

    """
    get_visualizer(Target.RVC4, dir_path).visualize()


@app.meta.command
def archive(
    path: str,
    *,
    save_path: str | None = None,
    put_file_plugin: str | None = None,
) -> None:
    """Convert a model file to a Luxonis NN Archive.

    Args:
        path: A URL or a path to the model file.
        save_path: Path or URL to save the archive to. By default, it is
            saved to the current directory under the name of the model.
        put_file_plugin: The name of the plugin to use for uploading the
            file.

    """
    model_path = resolve_path(path, MODELS_DIR)
    cfg = archive_from_model(model_path)
    save_path = save_path or f"{cfg.model.metadata.name}.tar.xz"
    if save_path.endswith("tar.xz"):
        compression = "xz"
    elif save_path.endswith("tar.gz"):
        compression = "gz"
    elif save_path.endswith("tar.bz2"):
        compression = "bz2"
    else:
        compression = "xz"

    if not save_path.endswith(f".tar.{compression}"):
        save_path += f"/{cfg.model.metadata.name}.tar.{compression}"
    archive_name = save_path.split("/")[-1]
    protocol = LuxonisFileSystem.get_protocol(save_path)
    if protocol != "file":
        archive_save_path = "./"
    else:
        archive_save_path = str(Path(save_path).parent)
    archive_save_path = ArchiveGenerator(
        archive_name=archive_name,
        save_path=archive_save_path,
        compression=compression,
        cfg_dict=cfg.model_dump(),
        executables_paths=[str(model_path)],
    ).make_archive()

    if protocol != "file":
        upload_to_remote(archive_save_path, save_path, put_file_plugin)
        Path(archive_save_path).unlink()
        logger.info(f"Archive uploaded to {save_path}")
    else:
        logger.info(f"Archive saved to {save_path}")


cache_app = App(
    name="cache",
    help="Manage the hidden modelconverter cache "
    "(staged inputs and remote downloads).",
)
app.meta.command(cache_app)


_CACHE_SUBDIR_DESCRIPTIONS = {
    "inputs": "Staged input files",
    "models": "Downloaded models",
    "calibration_data": "Calibration datasets",
    "misc": "Miscellaneous downloads",
    "configs": "Configuration files",
}


def _cache_entries(root: Path) -> list[Path] | None:
    """Return the cache's top-level entries, or ``None`` if the cache
    root itself cannot be read.

    A container killed before its entrypoint could chown the mounts back
    can leave the cache root owned by another user, which is precisely
    the situation these commands exist to report on and recover from.
    """
    if not root.exists():
        return []
    try:
        return sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        return None


@cache_app.command(name="info", sort_key=0)
def cache_info() -> None:
    """Report the location and disk usage of the modelconverter
    cache.
    """
    console = Console()
    root = get_cache_dir()

    entries = _cache_entries(root)
    if entries is None:
        console.print(
            Panel(
                f"The cache at [cyan]{root}[/] cannot be read. It is owned "
                "by another user, most likely written by a container that "
                "was killed before it could hand it back.\nRemove it with "
                f"[bold]sudo rm -rf {root}[/].",
                title="[bold]ModelConverter cache[/]",
                border_style="yellow",
                expand=False,
            )
        )
        return

    if not entries:
        console.print(
            Panel(
                f"The cache is empty.\nLocation: [cyan]{root}[/]",
                title="[bold]ModelConverter cache[/]",
                border_style="cyan",
                expand=False,
            )
        )
        return

    table = Table(box=box.SIMPLE_HEAD, expand=False, pad_edge=False)
    table.add_column("Directory", style="cyan", no_wrap=True)
    table.add_column("Contents", style="dim")
    table.add_column("Files", justify="right", style="magenta")
    table.add_column("Size", justify="right", style="green")

    total_size = 0
    total_files = 0
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            size, count = dir_stats(entry)
            description = _CACHE_SUBDIR_DESCRIPTIONS.get(entry.name, "")
        else:
            try:
                size, count = entry.stat().st_size, 1
            except OSError:
                size, count = 0, 1
            description = ""
        total_size += size
        total_files += count
        table.add_row(entry.name, description, str(count), human_size(size))

    table.add_section()
    table.add_row(
        "[bold]Total[/]",
        "",
        f"[bold]{total_files}[/]",
        f"[bold]{human_size(total_size)}[/]",
    )

    budget = cache_budget()
    if budget:
        table.add_row(
            "[dim]Budget[/]",
            "[dim]MODELCONVERTER_CACHE_MAX_SIZE[/]",
            "",
            f"[dim]{human_size(budget)}[/]",
        )

    console.print(
        Panel(
            table,
            title="[bold]ModelConverter cache[/]",
            subtitle=f"[dim]{root}[/]",
            border_style="cyan",
            expand=False,
        )
    )


def _refuse_while_in_use(console: Console, root: Path) -> bool:
    """Whether a running conversion still needs the cache.

    The cache is bind-mounted into every running container, so emptying it
    would pull the staged inputs -- and the downloads the container writes
    as it runs -- out from under a conversion that is still using them.
    """
    if not cache_is_in_use():
        return False
    console.print(
        f"Not clearing [cyan]{root}[/]: it is still in use by a running "
        "conversion. Wait for it to finish and try again."
    )
    return True


def _confirm(question: str) -> bool:
    """Ask ``question``, treating anything but an answer as a decline.

    Piped or closed stdin (a CI step, ``</dev/null``) makes ``input()``
    raise ``EOFError``, and Ctrl-C at the prompt raises
    ``KeyboardInterrupt``. A destructive command must not abort with a
    traceback on either, nor read silence for consent nobody gave.
    """
    try:
        return Confirm.ask(question)
    except (EOFError, KeyboardInterrupt):
        return False


@cache_app.command(name="clean", sort_key=1)
def cache_clean(
    *,
    yes: Annotated[
        bool,
        Parameter(
            name="-y",
            negative_bool=[],
        ),
    ] = False,
) -> None:
    """Remove the entire modelconverter cache.

    Args:
        yes: Clear the cache without prompting for confirmation.

    """
    console = Console()
    root = get_cache_dir()
    entries = _cache_entries(root)
    if entries is not None and not entries:
        console.print(f"Cache is already empty ([cyan]{root}[/]).")
        return

    if _refuse_while_in_use(console, root):
        return
    size, _ = dir_stats(root)
    if not yes and not _confirm(
        f"Clear the entire ModelConverter cache at [cyan]{root}[/]?"
    ):
        console.print("Cache clean cancelled.")
        return

    if _refuse_while_in_use(console, root):
        return

    shutil.rmtree(root, ignore_errors=True)
    if root.exists():
        remaining, _ = dir_stats(root)
        console.print(
            f"Cleared what could be removed from [cyan]{root}[/] "
            f"([green]{human_size(size - remaining)}[/] freed, "
            f"[yellow]{human_size(remaining)}[/] left). The remaining "
            "files are owned by another user, most likely written by a "
            "container that was killed before it could hand them back. "
            f"Remove them with [bold]sudo rm -rf {root}[/]."
        )
        return
    console.print(
        f":wastebasket:  Cleared cache at [cyan]{root}[/] "
        f"(freed [green]{human_size(size)}[/])."
    )


@app.meta.default
def launcher(
    *tokens: Annotated[
        str,
        Parameter(
            show=False,
            allow_leading_hyphen=True,
            json_dict=False,
            json_list=False,
        ),
    ],
    dev: Annotated[bool, Parameter(group=docker_parameters)] = False,
    gpu: Annotated[bool, Parameter(group=docker_parameters)] = True,
    tool_version: Annotated[
        str | None, Parameter(group=docker_parameters)
    ] = None,
    image: Annotated[
        str | None,
        Parameter(["image", "docker-image"], group=docker_parameters),
    ] = None,
    memory: Annotated[str | None, Parameter(group=docker_parameters)] = None,
    cpus: Annotated[float | None, Parameter(group=docker_parameters)] = None,
):
    """Run a command, in a docker container unless already inside one.

    Args:
        *tokens: The command and its arguments, parsed by the wrapped
            app.
        dev: If ``True``, builds and runs the target's ``dev`` image,
            which also mounts the host's sources, tests and
            ``pyproject.toml`` over the ones baked into it.
        gpu: If ``True``, runs the container with the ``nvidia``
            runtime. Only has an effect for the ``hailo`` target.
        tool_version: Version of the underlying conversion tools to use.
            Available options differ based on the target platform.
        image: Full name of the docker image to use. If the name
            includes a tag (e.g. ``:latest``), it will be used as is and
            the ``--tool-version`` argument will be ignored.
        memory: Amount of memory to allocate to the docker container, as
            a number with an optional binary unit: ``4g`` for four
            gibibytes, ``512m``, ``2GiB``, or a bare count of bytes. By
            default, uses all available system memory.
        cpus: Number of CPU cores to allocate to the docker container.
            Can be a fractional number, e.g. ``0.5`` for half a core. By
            default, uses all available CPU cores.

    """
    command, bound, _ = app.parse_args(tokens)
    target = bound.arguments.get("target")
    is_convert_command = getattr(command, "__name__", "") == "convert"
    running_in_docker = in_docker()

    memory_bytes = parse_size(memory) if memory is not None else None
    if memory_bytes is not None and memory_bytes <= 0:
        raise ValueError("Memory value must be a positive size.")

    if cpus is not None and cpus <= 0:
        raise ValueError("CPUs value must be a positive number.")

    def run_in_configured_environment() -> Any:
        if running_in_docker:
            return command(*bound.args, **bound.kwargs)

        assert target is not None
        tag = "dev" if dev else "latest"
        if dev:
            version = tool_version or get_default_target_version(target.value)
            if not (
                os.getenv("CI") == "true"
                and get_local_docker_image(
                    target.value,
                    bare_tag=tag,
                    version=version,
                    image=image,
                )
            ):
                docker_build(
                    target.value, bare_tag=tag, version=version, image=image
                )

        staged_tokens = stage_inputs(list(tokens), path_flags_for(command))

        docker_exec(
            target.value,
            *staged_tokens,
            bare_tag=tag,
            use_gpu=gpu,
            version=tool_version,
            image=image,
            memory=memory_bytes,
            cpus=cpus,
        )
        return None

    if not is_convert_command:
        return run_in_configured_environment()

    if running_in_docker:
        return run_in_configured_environment()

    assert target is not None
    command_telemetry = get_component_telemetry()
    previous_conversion_run_id = os.environ.get(CONVERSION_RUN_ID_ENV_VAR)
    conversion_run_id = get_conversion_run_id()
    command_start = time.monotonic()
    caught_exc: BaseException | None = None

    try:
        return run_in_configured_environment()
    except BaseException as exc:
        caught_exc = exc
        raise
    finally:
        try:
            command_telemetry.capture(
                COMMAND_EVENT,
                build_command_properties(
                    conversion_run_id=conversion_run_id,
                    target=target,
                    runs_in_docker=not running_in_docker,
                    dev_image=dev,
                    gpu_enabled=gpu,
                    target_tool_version=resolve_target_tool_version(
                        target=target,
                        tool_version=tool_version,
                        image=image,
                    ),
                    custom_image_provided=image is not None,
                    memory_limit_set=memory is not None,
                    cpu_limit_set=cpus is not None,
                    result=command_result_from_exception(caught_exc),
                    failure_reason=command_failure_reason_from_exception(
                        caught_exc
                    ),
                    duration_ms=int((time.monotonic() - command_start) * 1000),
                ),
                include_system_metadata=True,
                distinct_id=conversion_run_id,
            )
        finally:
            if previous_conversion_run_id is None:
                os.environ.pop(CONVERSION_RUN_ID_ENV_VAR, None)
            else:
                os.environ[CONVERSION_RUN_ID_ENV_VAR] = (
                    previous_conversion_run_id
                )


if __name__ == "__main__":
    app.meta()
