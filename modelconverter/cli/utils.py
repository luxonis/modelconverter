import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from luxonis_ml.nn_archive import is_nn_archive
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import PreprocessingBlock

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
