import re
import shutil
import sys
from contextlib import suppress
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import sleep
from typing import Any, Literal
from uuid import UUID

import typer
from loguru import logger
from luxonis_ml.nn_archive import is_nn_archive
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import PreprocessingBlock
from luxonis_ml.typing import Params
from packaging.version import Version
from requests.exceptions import HTTPError
from rich import print
from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import Progress
from rich.table import Table

from modelconverter.hub.hub_requests import Request
from modelconverter.utils import (
    process_nn_archive,
    resolve_path,
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


class ModelType(str, Enum):
    ONNX = "ONNX"
    IR = "IR"
    PYTORCH = "PYTORCH"
    TFLITE = "TFLITE"
    RVC2 = "RVC2"
    RVC3 = "RVC3"
    RVC4 = "RVC4"
    HAILO = "HAILO"

    @classmethod
    def from_suffix(cls, suffix: str) -> "ModelType":
        if suffix == ".onnx":
            return cls.ONNX
        if suffix == ".tflite":
            return cls.TFLITE
        if suffix in [".xml", ".bin"]:
            return cls.IR
        if suffix in [".pt", ".pth"]:
            return cls.PYTORCH
        raise ValueError(f"Unsupported model format: {suffix}")


def get_output_dir_name(
    target: Target, name: str, output_dir: str | None
) -> Path:
    date = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    if output_dir is not None:
        if (OUTPUTS_DIR / output_dir).exists():
            shutil.rmtree(OUTPUTS_DIR / output_dir)
    else:
        output_dir = f"{name}_to_{target.name.lower()}_{date}"
    return OUTPUTS_DIR / output_dir


def init_dirs() -> None:
    for p in [CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, CALIBRATION_DIR]:
        logger.debug(f"Creating {p}")
        p.mkdir(parents=True, exist_ok=True)


def get_configs(
    path: str | None, opts: list[str] | dict[str, Any] | None = None
) -> tuple[Config, NNArchiveConfig | None, str | None]:
    """Sets up the configuration.

    @type path: Optional[str]
    @param path: Path to the configuration file or NN Archive.
    @type opts: Optional[List[str]]
    @param opts: Optional CLI overrides of the config file.
    @rtype: Tuple[Config, Optional[NNArchiveConfig], Optional[str]]
    @return: Tuple of the parsed modelconverter L{Config},
        L{NNArchiveConfig} and the main stage key.
    """

    opts = opts or []
    if isinstance(opts, list):
        if len(opts) % 2 != 0:
            raise ValueError(
                "Invalid number of overrides. See --help for more information."
            )
        overrides: Params = {
            opts[i]: opts[i + 1] for i in range(0, len(opts), 2)
        }
    else:
        overrides = opts
    if path is not None:
        path_ = resolve_path(path, MISC_DIR)
        if path_.is_dir() or is_nn_archive(path_):
            return process_nn_archive(path_, overrides)
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
) -> tuple[Config, dict[str, PreprocessingBlock]]:
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


def print_hub_resource_info(
    model: dict[str, Any],
    keys: list[str],
    json: bool,
    rename: dict[str, str] | None = None,
    **kwargs,
) -> None:
    rename = rename or {}

    if json:
        print(model)
        return

    console = Console()

    if model.get("description_short"):
        description_short_panel = Panel(
            f"[italic]{model['description_short']}[/italic]",
            border_style="dim",
            box=ROUNDED,
        )
    else:
        description_short_panel = None

    table = Table(show_header=False, box=None)
    table.add_column(justify="right", style="bold")
    table.add_column()
    for key in keys:
        if key in ["created", "updated", "last_version_added"]:
            value = model.get(key, "N/A")

            def format_date(date_str: str) -> RenderableType:
                try:
                    date_obj = datetime.strptime(
                        date_str, "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    return date_obj.strftime("%B %d, %Y %H:%M:%S")
                except (ValueError, TypeError):
                    return Pretty("N/A")

            formatted_value = format_date(value)

            table.add_row(
                f"{rename.get(key, key).replace('_', ' ').title()}:",
                formatted_value,
            )
        elif key == "is_public":
            if key not in model:
                value = "N/A"
            else:
                value = "Public" if model[key] else "Private"
            table.add_row("Visibility", Pretty(value))
        elif key == "is_commercial":
            # TODO: Is Usage the right term?
            if key not in model:
                value = "N/A"
            else:
                value = "Commercial" if model[key] else "Non-Commercial"
            table.add_row("Usage:", Pretty(value))
        elif key == "is_nn_archive":
            table.add_row("NN Archive:", Pretty(model.get(key, False)))
        else:
            table.add_row(
                f"{rename.get(key, key).replace('_', ' ').title()}:",
                Pretty(model.get(key, "N/A")),
            )

    info_panel = Panel(table, border_style="cyan", box=ROUNDED, **kwargs)

    if model.get("description"):
        description_panel = Panel(
            Markdown(model["description"]),
            title="Description",
            border_style="green",
            box=ROUNDED,
        )
    else:
        description_panel = None

    nested_panels = []
    if description_short_panel:
        nested_panels.append(description_short_panel)
    nested_panels.append(info_panel)
    if description_panel:
        nested_panels.append(description_panel)

    content = Group(*nested_panels)

    main_panel = Panel(
        content,
        title=f"[bold magenta]{model.get('name', 'N/A')}[/bold magenta]",
        width=74,
        border_style="magenta",
        box=ROUNDED,
    )

    console.print(main_panel)


def hub_ls(
    endpoint: str,
    keys: list[str],
    rename: dict[str, str] | None = None,
    *,
    _silent: bool = False,
    **kwargs,
) -> list[dict[str, Any]]:
    rename = rename or {}
    data = Request.get(f"{endpoint}/", params=kwargs)
    table = Table(row_styles=["yellow", "cyan"], box=ROUNDED)
    for key in keys:
        table.add_column(rename.get(key, key), header_style="magenta i")

    for model in data:
        renderables = []
        for key in keys:
            value = str(model.get(key, "N/A"))
            if isinstance(value, list):
                value = ", ".join(value)
            renderables.append(value)
        table.add_row(*renderables)

    if not _silent:
        console = Console()
        console.print(table)
    return data


def is_valid_uuid(uuid_string: str) -> bool:
    try:
        UUID(uuid_string)
    except Exception:
        return False
    else:
        return True


def slug_to_id(
    slug: str, endpoint: Literal["models", "modelVersions", "modelInstances"]
) -> str:
    for is_public in [True, False]:
        with suppress(HTTPError):
            params = {
                "is_public": is_public,
                "slug": slug,
            }
            data = Request.get(f"{endpoint}/", params=params)
            if data:
                return data[0]["id"]
    raise ValueError(f"Model with slug '{slug}' not found.")


def get_resource_id(
    identifier: str,
    endpoint: Literal["models", "modelVersions", "modelInstances"],
) -> str:
    if is_valid_uuid(identifier):
        return identifier
    return slug_to_id(identifier, endpoint)


def request_info(
    identifier: str,
    endpoint: Literal["models", "modelVersions", "modelInstances"],
) -> dict[str, Any]:
    resource_id = get_resource_id(identifier, endpoint)

    try:
        return Request.get(f"{endpoint}/{resource_id}/")
    except HTTPError:
        typer.echo(f"Resource with ID '{resource_id}' not found.")
        sys.exit(1)


def get_variant_name(
    cfg: SingleStageConfig, model_type: ModelType, name: str
) -> str:
    shape = cfg.inputs[0].shape
    layout = cfg.inputs[0].layout

    if shape is not None:
        if layout is not None and "H" in layout and "W" in layout:
            h, w = shape[layout.index("H")], shape[layout.index("W")]
            return f"{name} {h}x{w}"
        if len(shape) == 4:
            if model_type == ModelType.TFLITE:
                h, w = shape[1], shape[2]
            else:
                h, w = shape[2], shape[3]
            return f"{name} {h}x{w}"
    return name


def get_version_number(model_id: str) -> str:
    versions = Request.get("modelVersions/", params={"model_id": model_id})
    if not versions:
        return "0.1.0"
    max_version = Version(versions[0]["version"])
    for v in versions[1:]:
        max_version = max(max_version, Version(v["version"]))
    max_version = str(max_version)
    version_numbers = max_version.split(".")
    version_numbers[-1] = str(int(version_numbers[-1]) + 1)
    return ".".join(version_numbers)


def wait_for_export(run_id: str) -> None:
    def _get_run(run_id: str) -> dict[str, Any]:
        return Request.dag_get(f"runs/{run_id}")

    def _clean_logs(logs: str) -> str:
        pattern = r"\[.*?\] \{.*?\} INFO - \[base\] logs:\s*"
        return re.sub(pattern, "", logs)

    with Progress() as progress:
        progress.add_task("Waiting for the conversion to finish", total=None)
        run = _get_run(run_id)
        while run["status"] in ["PENDING", "RUNNING"]:
            sleep(10)
            run = _get_run(run_id)

    if run["status"] == "FAILURE":
        if run["logs"] is None:
            print(run["id"])
            raise RuntimeError("Export failed with no logs.")

        if run["logs"] is None:
            raise RuntimeError("Export failed with no logs.")

        while len(run["logs"].split("\n")) < 5:
            run = _get_run(run_id)
            sleep(5)

        logs = _clean_logs(run["logs"])
        raise RuntimeError(f"Export failed with\n{logs}.")


def get_target_specific_options(
    target: Target, cfg: SingleStageConfig, tool_version: str | None = None
) -> dict[str, Any]:
    json_cfg = cfg.model_dump(mode="json")
    options = {
        "disable_onnx_simplification": cfg.disable_onnx_simplification,
        "disable_onnx_optimization": cfg.disable_onnx_optimization,
        "inputs": json_cfg["inputs"],
    }
    if target is Target.RVC4:
        options["snpe_onnx_to_dlc_args"] = cfg.rvc4.snpe_onnx_to_dlc_args
        options["snpe_dlc_quant_args"] = cfg.rvc4.snpe_dlc_quant_args
        options["snpe_dlc_graph_prepare_args"] = (
            cfg.rvc4.snpe_dlc_graph_prepare_args
        )
        if tool_version is not None:
            options["snpe_version"] = tool_version
    elif target in [Target.RVC2, Target.RVC3]:
        target_cfg = getattr(cfg, target.value)
        options["mo_args"] = target_cfg.mo_args
        options["compile_tool_args"] = target_cfg.compile_tool_args
        if tool_version is not None:
            options["ir_version"] = tool_version
        if target is Target.RVC3:
            options["pot_target_device"] = cfg.rvc3.pot_target_device
    elif target is Target.HAILO:
        options["optimization_level"] = cfg.hailo.optimization_level
        options["compression_level"] = cfg.hailo.compression_level
        options["batch_size"] = cfg.hailo.batch_size
        options["disable_calibration"] = cfg.hailo.disable_calibration

    return options
