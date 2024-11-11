import logging
import shutil
from datetime import datetime
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from luxonis_ml.nn_archive import ArchiveGenerator, is_nn_archive
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import PreprocessingBlock
from luxonis_ml.utils import LuxonisFileSystem, reset_logging, setup_logging
from typing_extensions import Annotated, TypeAlias

from modelconverter.packages import (
    get_benchmark,
    get_exporter,
    get_inferer,
)
from modelconverter.utils import (
    ModelconverterException,
    archive_from_model,
    docker_build,
    docker_exec,
    in_docker,
    modelconverter_config_to_nn,
    process_nn_archive,
    resolve_path,
    upload_file_to_remote,
)
from modelconverter.utils.config import Config, SingleStageConfig
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    CONFIGS_DIR,
    MISC_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from modelconverter.utils.types import DataType, Encoding, Target

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Modelconverter CLI",
    add_completion=False,
    rich_markup_mode="markdown",
)


class Format(str, Enum):
    NATIVE = "native"
    NN_ARCHIVE = "nn_archive"


FormatOption: TypeAlias = Annotated[
    Format,
    typer.Option(
        help="One of the supported formats.",
    ),
]
VersionOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(
        "-v",
        "--version",
        help="""Version of the underlying conversion tools to use.
        Available options differ based on the target platform:

          - `RVC2`:
            - `2021.4.0`
            - `2022.3.0` (default)

          - `RVC3`:
            - `2022.3.0` (default)

          - `RVC4`:
            - `2.23.0` (default)
            - `2.24.0`
            - `2.25.0`
            - `2.26.2`
            - `2.27.0`

          - `HAILO`:
              - `2024.04` (default),
              - `2024.07` (default)""",
        show_default=False,
    ),
]
PathOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(
        help="Path to the configuration file or nn archive.",
        show_default=False,
    ),
]
OptsArgument: TypeAlias = Annotated[
    Optional[List[str]],
    typer.Argument(
        help="A list of optional CLI overrides of the config file.",
        show_default=False,
    ),
]

TargetArgument: TypeAlias = Annotated[
    Target,
    typer.Argument(
        case_sensitive=False,
        help="Target platform to convert to.",
        show_default=False,
    ),
]

DevOption: TypeAlias = Annotated[
    bool,
    typer.Option(
        help="Builds a new image and uses the development docker-compose file."
    ),
]

ModelPathOption: TypeAlias = Annotated[
    str, typer.Option(help="A URL or a path to the model file.")
]

DockerOption: TypeAlias = Annotated[
    bool,
    typer.Option(
        help="Runs the conversion in a docker container. "
        "Ensure that all the necessary tools are available in "
        "PATH if you disable this option.",
    ),
]

GPUOption: TypeAlias = Annotated[
    bool,
    typer.Option(help="Use GPU for conversion. Only relevant for HAILO."),
]

OutputDirOption: TypeAlias = Annotated[
    Optional[str],
    typer.Option(
        ..., "--output-dir", "-o", help="Name of the output directory."
    ),
]


def get_output_dir_name(
    target: Target, name: str, output_dir: Optional[str]
) -> Path:
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if output_dir is not None:
        if (OUTPUTS_DIR / output_dir).exists():
            shutil.rmtree(OUTPUTS_DIR / output_dir)
    else:
        output_dir = f"{name}_to_{target.name.lower()}_{date}"
    return OUTPUTS_DIR / output_dir


def get_configs(
    path: Optional[str], opts: OptsArgument = None
) -> Tuple[Config, Optional[NNArchiveConfig], Optional[str]]:
    """Sets up the configuration.

    @type path: Optional[str]
    @param path: Path to the configuration file or NN Archive.
    @type opts: Optional[List[str]]
    @param opts: Optional CLI overrides of the config file.
    @rtype: Tuple[Config, Optional[NNArchiveConfig], Optional[str]]
    @return: Tuple of the parsed modelconverter L{Config}, L{NNArchiveConfig} and the
        main stage key.
    """

    for p in [CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, CALIBRATION_DIR]:
        logger.debug(f"Creating {p}")
        p.mkdir(parents=True, exist_ok=True)

    opts = opts or []
    if len(opts) % 2 != 0:
        raise ValueError(
            "Invalid number of overrides. See --help for more information."
        )
    overrides = {opts[i]: opts[i + 1] for i in range(0, len(opts), 2)}
    if path is not None:
        path_ = resolve_path(path, MISC_DIR)
        if path_.is_dir() or is_nn_archive(path_):
            return process_nn_archive(path_, overrides)
        shutil.move(str(path_), CONFIGS_DIR / path_.name)
    cfg = Config.get_config(path, overrides)

    main_stage_key = None
    if len(cfg.stages) > 1:
        for key in cfg.stages:
            if "yolov8" in key and "seg" in key:
                logger.info(f"Detected main stage key: {key}")
                main_stage_key = key
                break
    else:
        main_stage_key = next(iter(cfg.stages.keys()))

    return cfg, None, main_stage_key


def extract_preprocessing(
    cfg: Config,
) -> Tuple[Config, Dict[str, PreprocessingBlock]]:
    if len(cfg.stages) > 1:
        raise ValueError(
            "Only single-stage models are supported with NN archive."
        )
    stage_cfg = next(iter(cfg.stages.values()))
    preprocessing = {}
    for inp in stage_cfg.inputs:
        mean = inp.mean_values or [0, 0, 0]
        scale = inp.scale_values or [1, 1, 1]
        encoding = inp.encoding
        layout = inp.layout

        dai_type = encoding.to.value
        if dai_type != "NONE":
            if inp.data_type == DataType.FLOAT16:
                type = "F16F16F16"
            else:
                type = "888"
            dai_type += type
            dai_type += "i" if layout == "NHWC" else "p"

        preproc_block = PreprocessingBlock(
            mean=mean,
            scale=scale,
            reverse_channels=encoding.to == Encoding.RGB,
            interleaved_to_planar=layout == "NHWC",
            dai_type=dai_type,
        )
        preprocessing[inp.name] = preproc_block

        inp.mean_values = None
        inp.scale_values = None
        inp.encoding.from_ = Encoding.NONE
        inp.encoding.to = Encoding.NONE

    return cfg, preprocessing


@app.command()
def infer(
    target: TargetArgument,
    model_path: ModelPathOption,
    input_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--input-path",
            "-i",
            help="Path to the directory with data for inference."
            "The directory must contain one subdirectory per input, named the same as the input."
            "Inference data must be provided in the NPY format.",
        ),
    ],
    path: PathOption,
    output_dir: OutputDirOption,
    stage: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--stage",
            "-s",
            help="Name of the stage to run. Only needed for multistage configs.",
        ),
    ] = None,
    dev: DevOption = False,
    version: VersionOption = None,
    gpu: Annotated[
        bool,
        typer.Option(help="Use GPU for conversion. Only relevant for HAILO."),
    ] = True,
    opts: OptsArgument = None,
):
    """Runs inference on the specified target platform."""

    tag = "dev" if dev else "latest"

    if in_docker():
        setup_logging(file="modelconverter.log", use_rich=True)
        logger = logging.getLogger(__name__)
        logger.info("Starting inference")
        try:
            mult_cfg, _, _ = get_configs(path, opts)
            cfg = mult_cfg.get_stage_config(stage)
            Inferer = get_inferer(target)
            assert output_dir is not None
            Inferer.from_config(
                model_path, input_path, Path(output_dir), cfg
            ).run()
        except Exception:
            logger.exception("Encountered an unexpected error!")
            exit(2)
    else:
        if dev:
            docker_build(target.value, bare_tag=tag, version=version)
        args = [
            "infer",
            target.value,
            "--model-path",
            str(model_path),
            "--input-path",
            str(input_path),
            "--path",
            str(path),
        ]
        if output_dir is not None:
            args.extend(["--output-dir", output_dir])
        if opts is not None:
            args.extend(opts)
        docker_exec(
            target.value, *args, bare_tag=tag, use_gpu=gpu, version=version
        )


@app.command()
def shell(
    target: TargetArgument,
    dev: DevOption = False,
    version: VersionOption = None,
    gpu: GPUOption = True,
):
    """Boots up a shell inside a docker container for the specified target platform."""
    if dev:
        docker_build(target.value, bare_tag="dev", version=version)
    docker_exec(
        target.value,
        bare_tag="dev" if dev else "latest",
        version=version,
        use_gpu=gpu,
    )


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def benchmark(
    target: TargetArgument,
    model_path: ModelPathOption,
    ctx: typer.Context,
    full: Annotated[
        bool,
        typer.Option(
            ..., help="Runs the full benchmark using all configurations."
        ),
    ] = False,
    save: Annotated[
        bool, typer.Option(..., help="Saves the benchmark results to a file.")
    ] = False,
):
    """Runs benchmark on the specified target platform.

    Specific target options:




    **RVC2**

    - `--repetitions`: The number of repetitions to perform. Default: `1`

    - `--num-threads`: The number of threads to use for inference. Default: `2`

    ---

    **RVC3**

    - `--requests`: The number of requests to perform. Default: `1`

    ---

    **RVC4**

    - `--profile`: The SNPE profile to use for inference. Default: `"default"`

    - `--num-images`: The number of images to use for inference. Default: `1000`

    ---
    """

    setup_logging(use_rich=True)
    kwargs = {}
    for key, value in zip(ctx.args[::2], ctx.args[1::2]):
        if key.startswith("--"):
            key = key[2:].replace("-", "_")
        else:
            raise typer.BadParameter(f"Unknown argument: {key}")
        kwargs[key] = value
    Benchmark = get_benchmark(target)
    benchmark = Benchmark(str(model_path))
    benchmark.run(full=full, save=save, **kwargs)


@app.command()
def convert(
    target: TargetArgument,
    path: PathOption = None,
    output_dir: OutputDirOption = None,
    dev: DevOption = False,
    to: FormatOption = Format.NATIVE,
    gpu: GPUOption = True,
    version: VersionOption = None,
    main_stage: Annotated[
        Optional[str],
        typer.Option(
            ...,
            "--main-stage",
            "-m",
            help="Name of the stage with the main model. "
            "Only needed for multistage configs and when converting to NN Archive. "
            "When converting fron NN Archive, the stage names are named the "
            "same as the model files without the suffix.",
        ),
    ] = None,
    archive_preprocess: Annotated[
        bool,
        typer.Option(
            help="Add the pre-processing to the NN archive instead of the model. "
            "In case of conversion from archive to archive, it moves the "
            "preprocessing to the new archive.",
        ),
    ] = False,
    opts: OptsArgument = None,
):
    """Exports the model for the specified target platform."""

    tag = "dev" if dev else "latest"

    if archive_preprocess and to != Format.NN_ARCHIVE:
        raise ValueError(
            "--archive-preprocess can only be used with --to nn_archive"
        )

    setup_logging(use_rich=True)
    if in_docker():
        logger = logging.getLogger(__name__)
        try:
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
            reset_logging()
            setup_logging(
                file=str(output_path / "modelconverter.log"), use_rich=True
            )
            if is_multistage:
                from modelconverter.packages.multistage_exporter import (
                    MultiStageExporter,
                )

                exporter = MultiStageExporter(
                    target=target, config=cfg, output_dir=output_path
                )
            else:
                exporter = get_exporter(target)(
                    config=next(iter(cfg.stages.values())),
                    output_dir=output_path,
                )

            out_models = exporter.run()
            if not isinstance(out_models, list):
                out_models = [out_models]
            if to == Format.NN_ARCHIVE:
                from modelconverter.packages.base_exporter import Exporter

                logger.info("Converting to NN archive")
                assert main_stage is not None
                if len(out_models) > 1:
                    model_name = f"{main_stage}{out_models[0].suffix}"
                else:
                    model_name = out_models[0].name
                nn_archive = modelconverter_config_to_nn(
                    cfg,
                    Path(model_name),
                    archive_cfg,
                    preprocessing,
                    main_stage,
                    (
                        exporter.inference_model_path
                        if isinstance(exporter, Exporter)
                        else exporter.exporters[
                            main_stage
                        ].inference_model_path
                    ),
                )
                generator = ArchiveGenerator(
                    archive_name=f"{cfg.name}.{target.value.lower()}",
                    save_path=str(output_path),
                    cfg_dict=nn_archive.model_dump(),
                    executables_paths=[
                        str(out_model) for out_model in out_models
                    ]
                    + [str(output_path / "buildinfo.json")],
                )
                out_models = [generator.make_archive()]
                logger.info(f"Model exported to {out_models[0]}")

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
        except ModelconverterException:
            logger.exception(
                "Encountered an exception in the conversion process!"
            )
            exit(1)
        except Exception:
            logger.exception("Encountered an unexpected error!")
            exit(2)
    else:
        if dev:
            docker_build(target.value, bare_tag=tag, version=version)

        args = [
            "convert",
            target.value,
            "--to",
            to.value,
            "--archive-preprocess"
            if archive_preprocess
            else "--no-archive-preprocess",
        ]
        if main_stage is not None:
            args.extend(["--main-stage", main_stage])
        if output_dir is not None:
            args.extend(["--output-dir", output_dir])
        if path is not None:
            args.extend(["--path", path])
        if opts is not None:
            args.extend(opts)
        docker_exec(
            target.value, *args, bare_tag=tag, use_gpu=gpu, version=version
        )


@app.command()
def archive(
    path: Annotated[
        str, typer.Argument(help="Path or an URL of the model file.")
    ],
    save_path: Annotated[
        Optional[str],
        typer.Option(
            "-s",
            "--save-path",
            help="Path or URL to save the archive to. "
            "By default, it is saved to the current directory "
            "under the name of the model.",
        ),
    ] = None,
    put_file_plugin: Annotated[
        Optional[str],
        typer.Option(
            help="The name of the plugin to use for uploading the file."
        ),
    ] = None,
):
    setup_logging(use_rich=True)
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


def version_callback(value: bool):
    if value:
        typer.echo(f"ModelConverter Version: {version('modelconv')}")
        raise typer.Exit()


@app.callback()
def common(
    _: Annotated[
        bool,
        typer.Option(
            "-v",
            "--version",
            callback=version_callback,
            help="Show version and exit.",
        ),
    ] = False,
):
    pass


if __name__ == "__main__":
    app()
