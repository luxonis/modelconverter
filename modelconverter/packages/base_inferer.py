import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from typing_extensions import Self

from modelconverter.utils import resolve_path
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.types import DataType, Encoding, ResizeMethod


@dataclass
class Inferer(ABC):
    model_path: Path
    src: Path
    dest: Path
    in_shapes: dict[str, list[int]]
    in_dtypes: dict[str, DataType]
    out_shapes: dict[str, list[int]]
    out_dtypes: dict[str, DataType]
    resize_method: dict[str, ResizeMethod]
    encoding: dict[str, Encoding]
    config: SingleStageConfig | None = None

    def __post_init__(self):
        if self.dest.exists():
            logger.debug(f"Removing existing directory {self.dest}.")
            shutil.rmtree(self.dest)
        self.dest.mkdir(parents=True, exist_ok=True)
        self.setup()

    @classmethod
    def from_config(
        cls, model_path: str, src: Path, dest: Path, config: SingleStageConfig
    ) -> Self:
        for container, typ_name in [
            (config.inputs, "input"),
            (config.outputs, "output"),
        ]:
            for node in container:
                if node.shape is None:
                    raise ValueError(
                        f"Shape for {typ_name} '{node.name}' must be provided."
                    )

        return cls(
            model_path=resolve_path(model_path, Path.cwd()),
            src=src,
            dest=dest,
            in_shapes={inp.name: inp.shape for inp in config.inputs},  # type: ignore
            in_dtypes={inp.name: inp.data_type for inp in config.inputs},
            out_shapes={out.name: out.shape for out in config.outputs},  # type: ignore
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
            config=config,
        )

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]: ...

    def run(self) -> None:
        t = time.time()
        logger.info(f"Starting inference on {self.src}.")
        iterators = [input_name.iterdir() for input_name in self.src.iterdir()]
        for input_files in zip(*iterators, strict=True):
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
