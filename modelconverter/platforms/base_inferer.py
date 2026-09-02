"""Abstract base for running inference with a converted model.

`Inferer` implements the platform-independent half of the ``infer``
command: it pairs up the files found under the source directory,
invokes the model once per set of inputs and stores the raw outputs as
``.npy`` files. Every platform -- RVC2, RVC3, RVC4 and Hailo --
subclasses it inside its own Docker image and supplies only the
runtime setup and a single inference step.
"""

import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from typing_extensions import Self

from modelconverter.utils import ModelconverterException, resolve_path
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.constants import INFERENCE_MARKER
from modelconverter.utils.types import DataType, Encoding, ResizeMethod


@dataclass
class Inferer(ABC):
    """Platform-independent base for running a converted model.

    Instances are usually created with `Inferer.from_config`. The
    output directory and the runtime are prepared during construction,
    after which `Inferer.run` performs the inference.

    Attributes:
        model_path: Path to the converted model to run.
        src: Directory holding one sub-directory of input files per
            model input, each named after that input.
        dest: Directory the outputs are written to, one sub-directory
            per model output.
        in_shapes: Shape of every model input, keyed by input name.
        in_dtypes: Data type of every model input, keyed by input
            name.
        out_shapes: Shape of every model output, keyed by output name.
        out_dtypes: Data type of every model output, keyed by output
            name.
        resize_method: How an image is resized to the shape of the
            input it is fed to, keyed by input name.
        encoding: Color encoding expected by every input, keyed by
            input name.
        layout: Layout of an input, keyed by input name. Inputs with
            no known layout map to ``None``.
        config: Configuration the inferer was created from, if any.

    """

    model_path: Path
    src: Path
    dest: Path
    in_shapes: dict[str, list[int]]
    in_dtypes: dict[str, DataType]
    out_shapes: dict[str, list[int]]
    out_dtypes: dict[str, DataType]
    resize_method: dict[str, ResizeMethod]
    encoding: dict[str, Encoding]
    layout: dict[str, str | None] = field(default_factory=dict)
    config: SingleStageConfig | None = None

    def __post_init__(self):
        """Prepare the output directory and set up the runtime.

        The recreated directory is tagged with a marker file, which is
        how a later run recognizes the results as its own.
        `Inferer.setup` is called last.

        .. warning::
            An existing destination directory is deleted before the
            run.

        Raises:
            ModelconverterException: If the destination exists but is
                not a directory, or is a non-empty directory that does
                not hold inference results.

        """
        if self.dest.exists():
            if not self.dest.is_dir():
                raise ModelconverterException(
                    f"Refusing to overwrite '{self.dest}': it is not a "
                    "directory. Pass a different `--output-dir`."
                )
            if not (self.dest / INFERENCE_MARKER).exists() and any(
                self.dest.iterdir()
            ):
                raise ModelconverterException(
                    f"Refusing to overwrite '{self.dest}': it is not empty "
                    "and does not hold inference results. Pass a different "
                    "`--output-dir`."
                )
            logger.debug(f"Removing existing directory {self.dest}.")
            shutil.rmtree(self.dest)
        self.dest.mkdir(parents=True, exist_ok=True)
        (self.dest / INFERENCE_MARKER).touch()
        self.setup()

    @classmethod
    def from_config(
        cls, model_path: str, src: Path, dest: Path, config: SingleStageConfig
    ) -> Self:
        """Create an inferer from a single-stage configuration.

        The shapes, data types, resize methods, encodings and layouts
        are taken from the configured inputs and outputs.

        Args:
            model_path: Path to the converted model, resolved relative
                to the current working directory.
            src: Directory holding the input files.
            dest: Directory the outputs are written to.
            config: Configuration of the stage to run.

        Returns:
            The constructed inferer.

        Raises:
            ValueError: If any configured input or output has no
                shape.

        """
        in_shapes: dict[str, list[int]] = {}
        out_shapes: dict[str, list[int]] = {}
        for container, shapes, typ_name in [
            (config.inputs, in_shapes, "input"),
            (config.outputs, out_shapes, "output"),
        ]:
            for node in container:
                if node.shape is None:  # pragma: no cover
                    raise ValueError(
                        f"Shape for {typ_name} '{node.name}' must be provided."
                    )
                shapes[node.name] = node.shape

        return cls(
            model_path=resolve_path(model_path, Path.cwd()),
            src=src,
            dest=dest,
            in_shapes=in_shapes,
            in_dtypes={inp.name: inp.data_type for inp in config.inputs},
            out_shapes=out_shapes,
            out_dtypes={out.name: out.data_type for out in config.outputs},
            resize_method={
                inp.name: inp.calibration.resize_method
                if isinstance(inp.calibration, ImageCalibrationConfig)
                else ResizeMethod.RESIZE
                for inp in config.inputs
            },
            encoding={inp.name: inp.encoding.to for inp in config.inputs},
            layout={inp.name: inp.layout for inp in config.inputs},
            config=config,
        )

    @abstractmethod
    def setup(self) -> None:
        """Initialize the platform-specific runtime.

        Called once during construction, before any inference.

        """

    @abstractmethod
    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]:
        """Run the model on a single set of inputs.

        Args:
            inputs: Path to the file holding the data for every model
                input, keyed by input name.

        Returns:
            The model outputs, keyed by output name.

        """

    def run(self) -> None:
        """Run the model over every input in the source directory.

        The per-input sub-directories of the source directory are
        walked in lockstep, so one file from each of them forms a set
        of inputs. Every output is saved as a ``.npy`` file under a
        sub-directory of the destination named after that output, and
        named after the first input file of the set.

        """
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
