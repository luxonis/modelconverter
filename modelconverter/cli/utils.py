import logging
import shutil
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import UUID

from luxonis_ml.nn_archive import is_nn_archive
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import PreprocessingBlock
from rich import print
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from modelconverter.hub.hub_requests import Request
from modelconverter.utils import (
    process_nn_archive,
    resolve_path,
)
from modelconverter.utils.config import Config
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    CONFIGS_DIR,
    MISC_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from modelconverter.utils.types import Target

logger = logging.getLogger(__name__)


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


def init_dirs() -> None:
    for p in [CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, CALIBRATION_DIR]:
        logger.debug(f"Creating {p}")
        p.mkdir(parents=True, exist_ok=True)


def get_configs(
    path: Optional[str], opts: Optional[List[str]] = None
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

        preproc_block = PreprocessingBlock(
            reverse_channels=inp.reverse_input_channels,
            mean=mean,
            scale=scale,
            interleaved_to_planar=False,
        )
        preprocessing[inp.name] = preproc_block

        inp.mean_values = None
        inp.scale_values = None
        inp.reverse_input_channels = False

    return cfg, preprocessing


def print_hub_resource_info(
    model: Dict[str, Any], keys: List[str], json: bool, **kwargs
) -> Dict[str, Any]:
    if json:
        print(model)
        return model

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

            def format_date(date_str):
                try:
                    date_obj = datetime.strptime(
                        date_str, "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    return date_obj.strftime("%B %d, %Y %H:%M:%S")
                except (ValueError, TypeError):
                    return Pretty("N/A")

            formatted_value = format_date(value)

            table.add_row(f"{key.replace('_', ' ').title()}:", formatted_value)
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
                f"{key.replace('_', ' ').title()}:",
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
    console.rule()
    return model


def hub_ls(endpoint: str, keys: List[str], **kwargs) -> None:
    data = Request.get(f"/api/v1/{endpoint}/", params=kwargs).json()
    print(kwargs)
    table = Table(row_styles=["yellow", "cyan"], box=ROUNDED)
    for key in keys:
        table.add_column(key, header_style="magenta i")

    for model in data:
        renderables = []
        for key in keys:
            value = model.get(key, "N/A")
            if isinstance(value, list):
                value = ", ".join(value)
            renderables.append(value)
        table.add_row(*renderables)

    console = Console()
    console.print(table)


def is_valid_uuid(uuid_string: str) -> bool:
    try:
        UUID(uuid_string)
        return True
    except Exception:
        return False


def slug_to_id(
    slug: str, endpoint: Literal["models", "modelVersions", "modelInstances"]
) -> str:
    for is_public in [True, False]:
        with suppress(Exception):
            params = {
                "is_public": is_public,
                "slug": slug,
            }
            data = Request.get(f"/api/v1/{endpoint}/", params=params).json()
            if data:
                return data[0]["id"]
    raise ValueError(f"Model with slug '{slug}' not found.")


def request_info(
    identifier: str,
    endpoint: Literal["models", "modelVersions", "modelInstances"],
) -> Dict[str, Any]:
    if is_valid_uuid(identifier):
        resource_id = identifier
    else:
        resource_id = slug_to_id(identifier, endpoint)

    return Request.get(f"/api/v1/{endpoint}/{resource_id}/").json()