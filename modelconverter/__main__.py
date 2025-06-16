import importlib.metadata
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from cyclopts import App, Group, Parameter
from loguru import logger
from luxonis_ml.nn_archive import ArchiveGenerator
from luxonis_ml.utils import LuxonisFileSystem, setup_logging

from modelconverter.cli import (
    extract_preprocessing,
    get_configs,
    get_output_dir_name,
    init_dirs,
)
from modelconverter.hub.__main__ import app as hub_app
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
    in_docker,
    resolve_path,
    upload_file_to_remote,
)
from modelconverter.utils.config import SingleStageConfig
from modelconverter.utils.constants import MODELS_DIR
from modelconverter.utils.nn_archive import generate_archive
from modelconverter.utils.types import Target

app = App(
    name="Modelconverter",
    version=lambda: f"ModelConverter v{importlib.metadata.version('modelconv')}",
)
app.meta.command(hub_app, name="hub")

app.meta.group_parameters = Group("Global Parameters", sort_key=0)
app["--help"].group = app.meta.group_parameters
app["--version"].group = app.meta.group_parameters

docker_parameters = Group.create_ordered(
    "Docker Parameters", help="Global parameters for all docker commands"
)
docker_commands = Group.create_ordered("Docker Commands")
device_commands = Group.create_ordered("Device Commands")

OptsType: TypeAlias = Annotated[
    list[str] | None, Parameter(json_list=False, json_dict=False)
]


@contextmanager
def catch_exceptions():
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
    opts: OptsType = None,
    /,
    *,
    path: str | None = None,
    output_dir: str | None = None,
    to: Literal["native", "nn_archive"] = "native",
    main_stage: str | None = None,
    archive_preprocess: bool = False,
) -> None:
    """Exports the model for the specified target platform.

    Parameters
    ----------
    target: Target
        The target platform to export the model for.
    opts: list[str] | None
        A list of optional CLI overrides for the configuration file.
    path: str | None
        A URL or a path to the configuration file or NN Archive.
    output_dir: str | None
        Name of the directory where the exported model will be saved.
    to: Literal["native", "nn_archive"]
        Whether to export the model to a simple model file or a Luxonis NN Archive.
    main_stage: str | None
        Name of the stage with the main model.
        Only needed for multistage configs and when converting to NN Archive.
        When converting fron NN Archive, the stage names are named the
        same as the model files without the suffix.
    archive_preprocess: bool
        Add the pre-processing to the NN archive instead of the model.
        In case of conversion from archive to archive, it moves the
        preprocessing to the new archive.
    """

    with catch_exceptions():
        init_dirs()
        cfg, archive_cfg, _main_stage = get_configs(path, opts)
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
        setup_logging(file=str(output_path / "modelconverter.log"))
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

        out_models = exporter.run()
        if not isinstance(out_models, list):
            out_models = [out_models]
        if to == "nn_archive":
            from modelconverter.packages.base_exporter import Exporter

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
                )
            ]

        if isinstance(exporter.config, SingleStageConfig):
            upload_url = exporter.config.output_remote_url
            put_file_plugin = exporter.config.put_file_plugin
        else:
            _cfg = next(iter(exporter.config.stages.values()))
            upload_url = _cfg.output_remote_url
            put_file_plugin = _cfg.put_file_plugin

        for model_path in out_models:
            if upload_url is not None:
                logger.info(f"Uploading {model_path} to {upload_url}")
                upload_file_to_remote(
                    model_path,
                    upload_url,
                    put_file_plugin,
                )

            logger.info("Conversion finished successfully")


@app.command(group=docker_commands)
def infer(
    target: Target,
    opts: OptsType = None,
    /,
    *,
    model_path: str,
    input_path: Path,
    output_dir: str,
    config: str | None = None,
    path: str | None = None,
    stage: str | None = None,
) -> None:
    """Runs inference on the specified target platform.

    Parameters
    ----------
    target : Target
        The target platform to run the inference on.
    model_path : str
        A URL or a path to the model file.
    input_path : Path
        Path to the directory with data for inference.
        The directory must contain one subdirectory per input, named the same as the input.
        Inference data must be provided in the NPY format.
    output_dir: str
        Name of the directory where the inference results will be saved.
    config: str | None
        A URL or a path to the configuration file.
    path: str | None
        An alias for ``config``. Deprecated.
    stage: str | None
        Name of the stage to run. Only needed for multistage configs.
        If not provided, the first stage will be used.
    """

    if path is not None:
        config = path
    setup_logging(file="modelconverter.log")
    logger.info("Starting inference")
    with catch_exceptions():
        mult_cfg, _, _ = get_configs(str(config), opts)
        cfg = mult_cfg.get_stage_config(stage)
        get_inferer(
            target, model_path, input_path, Path(output_dir), cfg
        ).run()


@app.command(group=docker_commands)
def shell(
    target: Target,
    /,
    *,
    command: Annotated[str | None, Parameter(name=["-c", "--command"])] = None,
) -> None:
    """Boots up a shell inside a docker container for the specified
    target platform.

    Parameters
    ----------
    target : Target
        The target platform.
    command : str
        The command to run in the shell. If not provided, a bash shell is started.
        If you want to run a command with arguments, use quotes around the command.
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
    num_threads: Annotated[int, Parameter(group=["RVC2", "RVC4"])] = 2,
    num_messages: Annotated[int, Parameter(group=["RVC2", "RVC4"])] = 50,
    requests: Annotated[int, Parameter(group="RVC3")] = 1,
    profile: Annotated[
        Literal[
            "low_balanced",
            "balanced",
            "default",
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
    ] = "default",
    runtime: Annotated[Literal["dsp", "cpu"], Parameter(group="RVC4")] = "dsp",
    num_images: Annotated[int, Parameter(group="RVC4")] = 1000,
    dai_benchmark: Annotated[bool, Parameter(group="RVC4")] = True,
    device_ip: Annotated[str | None, Parameter(group="RVC4")] = None,
) -> None:
    """Runs benchmark on the specified target platform.

    Parameters
    ----------
    target : Target
        The target platform to run the benchmark on.
    model_path : str
        A URL or a path to the model file.
    full : bool
        If ``True``, runs the full benchmark using all configurations.
    save : bool
        If ``True``, saves the benchmark results to a file.

    repetitions : int
        The number of repetitions to perform. Only relevant for DAI benchmark.
    num_threads : int
        The number of threads to use for inference. Only relevant for DAI benchmark.
    num_messages : int
        The number of messages to measure for each report. Only relevant for DAI benchmark.

    requests : int
        The number of requests to perform.

    profile : str
        The SNPE profile to use for inference.
    runtime : str
        The SNPE runtime to use for inference (dsp or cpu).
    num_images : int
        The number of images to use for inference.
    dai_benchmark : bool
        Whether to run the benchmark using the DAI V3. If False the SNPE tools are used.
    device_ip : str | None
        The IP address of the device to run the benchmark on. If not provided, the default device found by DAI will be used.
    """

    if target in {Target.RVC2, Target.RVC4}:
        kwargs = {
            "repetitions": repetitions,
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
            }
    elif target is Target.RVC3:
        kwargs = {
            "requests": requests,
        }
    get_benchmark(target, model_path).run(full=full, save=save, **kwargs)


# TODO: Specify device ID in case more than one device is connected
@app.meta.command(group=device_commands)
def analyze(
    *,
    dlc_model_path: str,
    onnx_model_path: str,
    image_dirs: Annotated[list[str], Parameter(negative_iterable=[])],
    analyze_outputs: bool = True,
    analyze_cycles: bool = True,
) -> None:
    """Runs layer and cycle analysis on the specified DLC model.

    Requires the RVC4 device to be connected and accessible using
    the ``adb`` command.

    Parameters
    ----------
    dlc_model_path : str
        The path to the DLC model file.

    onnx_model_path : str
        The path to the corresponding ONNX model file that was used for converting to DLC.

    image_dirs : list[str]
        A list of names and paths to directories with images for each input of the model.

    analyze_outputs : bool
        Whether to analyze the layer outputs.

    analyze_cycles : bool
        Whether to analyze the layer cycles.
    """

    with catch_exceptions():
        logger.info("Starting analysis")
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

        analyzer = get_analyzer(Target.RVC4, dlc_model_path, image_dirs_dict)
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
    """Visualizes the analysis results.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing the analysis results.
        The default search path is ``shared_with_container/outputs/analysis``.
    """
    get_visualizer(Target.RVC4, dir_path).visualize()


@app.meta.command
def archive(
    path: str,
    *,
    save_path: str | None = None,
    put_file_plugin: str | None = None,
) -> None:
    """Converts a model file to a Luxonis NN Archive.

    Parameters
    ----------
    path : str
        A URL or a path to the model file.
    save_path : str | None
        Path or URL to save the archive to. By default, it is saved to the current directory
        under the name of the model.
    put_file_plugin : str | None
        The name of the plugin to use for uploading the file.
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
        upload_file_to_remote(archive_save_path, save_path, put_file_plugin)
        Path(archive_save_path).unlink()
        logger.info(f"Archive uploaded to {save_path}")
    else:
        logger.info(f"Archive saved to {save_path}")


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
    dev: Annotated[
        bool,
        Parameter(
            group=docker_parameters,
            help="If ``True``, builds a new image and uses the development docker-compose file.",
        ),
    ] = False,
    gpu: Annotated[
        bool,
        Parameter(
            group=docker_parameters,
            help="If ``True``, uses the GPU version of the docker-compose file. ",
        ),
    ] = True,
    tool_version: Annotated[
        str | None,
        Parameter(
            group=docker_parameters,
            help="Version of the underlying conversion tools to use. "
            "Available options differ based on the target platform. ",
        ),
    ] = None,
):
    command, bound, _ = app.parse_args(tokens)

    if in_docker():
        return command(*bound.args, **bound.kwargs)

    tag = "dev" if dev else "latest"

    target = bound.arguments["target"]

    if dev:
        docker_build(target.value, bare_tag=tag, version=tool_version)

    docker_exec(
        target.value,
        *tokens,
        bare_tag=tag,
        use_gpu=gpu,
        version=tool_version,
    )


if __name__ == "__main__":
    app.meta()
