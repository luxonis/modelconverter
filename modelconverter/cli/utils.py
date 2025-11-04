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

from modelconverter.utils.hub_requests import Request
from modelconverter.utils import (
    process_nn_archive,
    resolve_path,
    sanitize_net_name,
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
    name = sanitize_net_name(name)
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