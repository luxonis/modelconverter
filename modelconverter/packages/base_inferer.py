import logging
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from modelconverter.utils import resolve_path
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.types import DataType, Encoding, ResizeMethod

logger = logging.getLogger(__name__)


@dataclass
class Inferer(ABC):
    model_path: Path
    src: Path
    dest: Path
    in_shapes: Dict[str, List[int]]
    in_dtypes: Dict[str, DataType]
    out_shapes: Dict[str, List[int]]
    out_dtypes: Dict[str, DataType]
    resize_method: Dict[str, ResizeMethod]
    encoding: Dict[str, Encoding]

    def __post_init__(self):
        if self.dest.exists():
            logger.debug(f"Removing existing directory {self.dest}.")
            shutil.rmtree(self.dest)
        self.dest.mkdir(parents=True, exist_ok=True)
        self.setup()

    @classmethod
    def from_config(
        cls, model_path: str, src: Path, dest: Path, config: SingleStageConfig
    ):
        return cls(
            model_path=resolve_path(model_path, Path.cwd()),
            src=src,
            dest=dest,
            in_shapes={inp.name: inp.shape for inp in config.inputs},
            in_dtypes={inp.name: inp.data_type for inp in config.inputs},
            out_shapes={out.name: out.shape for out in config.outputs},
            out_dtypes={out.name: out.data_type for out in config.outputs},
            resize_method={
                inp.name: inp.calibration.resize_method
                if isinstance(inp.calibration, ImageCalibrationConfig)
                else ResizeMethod.RESIZE
                for inp in config.inputs
            },
            encoding={
                inp.name: inp.encoding.to
                if isinstance(inp.calibration, ImageCalibrationConfig)
                else Encoding.BGR
                for inp in config.inputs
            },
        )

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def infer(self, inputs: Dict[str, Path]) -> Dict[str, np.ndarray]:
        pass

    def run(self):
        t = time.time()
        logger.info(f"Starting inference on {self.src}.")
        iterators = [input_name.iterdir() for input_name in self.src.iterdir()]
        for input_files in zip(*iterators):
            inputs = {
                file_path.parent.name: file_path for file_path in input_files
            }
            outputs = self.infer(inputs)

            for output_name, output in outputs.items():
                out_path = self.dest / output_name
                out_path.mkdir(parents=True, exist_ok=True)
                np.save(out_path / input_files[0].stem, output)
        logger.info(f"Inference finished in {time.time() - t} seconds.")
        logger.info(f"Inference results saved to {self.dest}.")
