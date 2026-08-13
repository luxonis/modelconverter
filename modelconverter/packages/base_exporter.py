"""Target-agnostic scaffolding for the model exporters.

Converting a model means handing it to a vendor toolchain that only
exists inside that target's Docker image, so what the exporters share is
everything around that call: staging the input model in the output
directory, optionally simplifying the ONNX, producing calibration data
for the inputs that ask for random ones, and recording what was run.
That part lives in `Exporter`, which the per-target exporters subclass.
"""

import json
import shutil
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from loguru import logger

from modelconverter.utils import (
    exit_with,
    read_calib_dir,
    sanitize_net_name,
    subprocess_run,
)
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    RandomCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.onnx_compatibility import (
    get_external_data_paths,
    has_external_data,
    save_onnx_model,
)
from modelconverter.utils.subprocess import SubprocessResult
from modelconverter.utils.types import InputFileType, Target


class Exporter(ABC):
    """Base class of the single-stage, per-target model exporters.

    Subclasses implement `export` to run their target's toolchain and
    `exporter_buildinfo` to describe what they ran; `run` wraps both
    with the shared bookkeeping.

    Attributes:
        target: The target platform the subclass converts for.

    """

    target: Target

    def __init__(
        self,
        config: SingleStageConfig,
        output_dir: Path,
    ):
        """Stage the input model and the calibration data.

        The model, along with any ONNX external data belonging to it, is
        copied into the output directory and the intermediate outputs
        directory, and simplified when the configuration asks for it.
        Unless the target is RVC2 or calibration is disabled, inputs
        configured for random calibration also get their data generated
        here, so that the export itself only ever sees image
        calibration.

        Args:
            config: Configuration of the single stage to convert.
            output_dir: Directory the converted model and the build
                information are written to.

        """
        input_model = config.input_model

        self.config = config
        self.output_dir = output_dir
        self.input_file_type = config.input_file_type
        self.inputs = {inp.name: inp for inp in config.inputs}
        self._inference_model_path: Path | None = None

        self.outputs = {out.name: out for out in config.outputs}
        self.keep_intermediate_outputs = config.keep_intermediate_outputs
        self.onnx_simplification = config.onnx_simplification
        self.onnx_optimizations = config.onnx_optimizations

        self.model_name = sanitize_net_name(input_model.stem)
        self.original_model_name = sanitize_net_name(
            input_model.name, with_suffix=True
        )

        self.intermediate_outputs_dir = (
            self.output_dir / "intermediate_outputs"
        )
        self.intermediate_outputs_dir.mkdir(parents=True, exist_ok=True)

        self._cmd_info: dict[str, list[str]] = {}
        self.is_tflite = self.input_file_type == InputFileType.TFLITE

        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(config.model_dump_json(indent=4))

        sanitized_model_name = (
            sanitize_net_name(input_model.stem) + input_model.suffix
        )
        shutil.copy(
            input_model,
            self.intermediate_outputs_dir / sanitized_model_name,
        )
        shutil.copy(input_model, self.output_dir / sanitized_model_name)
        # External data is an ONNX-only concept; loading a non-ONNX input
        # (e.g. an OpenVINO IR .xml/.bin) as ONNX would fail.
        external_data_paths = (
            get_external_data_paths(input_model)
            if self.input_file_type == InputFileType.ONNX
            else []
        )
        # A model saved with `all_tensors_to_one_file=False` has one companion
        # file per tensor, and each is located relative to the model, so the
        # layout has to be reproduced rather than flattened.
        # `get_external_data_paths` anchors the returned paths to the resolved
        # model *directory* -- not the directory of the resolved model, which
        # differs when the model itself is a symlink into another directory.
        model_dir = input_model.parent.resolve()
        for external_data_path in external_data_paths:
            relative = external_data_path.relative_to(model_dir)
            for directory in (self.intermediate_outputs_dir, self.output_dir):
                dest = directory / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(external_data_path, dest)
        if self.input_file_type == InputFileType.IR:
            assert self.config.input_bin is not None
            shutil.copy(
                self.config.input_bin,
                (
                    self.intermediate_outputs_dir
                    / sanitize_net_name(self.config.input_bin.stem)
                ).with_suffix(".bin"),
            )
            self.config.input_bin = (
                self.config.input_bin.parent
                / sanitize_net_name(self.config.input_bin.stem)
            ).with_suffix(".bin")
        self.input_model = self.intermediate_outputs_dir / sanitized_model_name

        if (
            self.onnx_simplification
            and self.input_file_type == InputFileType.ONNX
        ):
            self.input_model = self.simplify_onnx()

        self._disable_calibration = getattr(
            self.config, self.target.name.lower()
        ).disable_calibration

        if self.target != Target.RVC2 and self._disable_calibration:
            logger.warning("Calibration has been disabled.")
            logger.warning("The quantization step will be skipped.")

        if self.target != Target.RVC2 and not self._disable_calibration:
            self._prepare_random_calibration_data()

    @property
    def inference_model_path(self) -> Path:
        """Return the path of the model to run inference with.

        Raises:
            ValueError: If the export has not been run yet, so no such
                model exists.

        """
        if self._inference_model_path is None:  # pragma: no cover
            raise ValueError(
                "Inference model path not yet set. Export must be run first."
            )
        return self._inference_model_path

    def simplify_onnx(self) -> Path:  # pragma: no cover
        """Simplify the staged ONNX model.

        The backend is whichever one the ``onnx_simplification`` option
        names, ``onnxsim`` or ``onnxslim``. A missing backend, a failed
        simplification or a simplified model that does not check out are
        all only warned about, leaving the model as it was.

        Returns:
            Path of the simplified model, or the path of the staged
            model if the simplification did not happen.

        """
        logger.info("Simplifying ONNX.")
        try:
            if self.onnx_simplification == "onnxsim":
                from onnxsim import simplify

                logger.info("Using `onnxsim` for simplification.")
            elif self.onnx_simplification == "onnxslim":
                from onnxslim import slim

                logger.info("Using `onnxslim` for simplification.")

                def simplify(model: str) -> tuple[onnx.ModelProto, bool]:
                    slimmed = slim(onnx.load(model))
                    return slimmed, bool(slimmed)  # type: ignore

        except ImportError:
            backend = self.onnx_simplification
            logger.warning(
                f"`{backend}` not installed, proceeding without simplification."
                f"Please install it using `pip install {backend}`."
            )
            return self.input_model

        try:
            onnx_sim, check = simplify(str(self.input_model))
        except Exception as e:  # pragma: no cover
            logger.warning(
                f"Failed to simplify ONNX: {e}. Proceeding without simplification."
            )
            return self.input_model
        if not check:  # pragma: no cover
            logger.warning(
                "Provided ONNX could not be simplified. "
                "Proceeding without simplification."
            )
            return self.input_model
        logger.info("ONNX successfully simplified.")
        onnx_sim_path = self._attach_suffix(
            self.input_model, "simplified.onnx"
        )
        logger.info(f"Saving simplified ONNX to {onnx_sim_path}")
        save_onnx_model(
            onnx_sim,
            onnx_sim_path,
            save_as_external_data=has_external_data(self.input_model),
            location=f"{onnx_sim_path.name}_data",
        )
        return onnx_sim_path

    @abstractmethod
    def exporter_buildinfo(self) -> dict[str, Any]:
        """Describe the target-specific side of the conversion.

        Returns:
            Whatever the target records about the tools it used, merged
            into ``buildinfo.json`` by `run`.

        """

    @abstractmethod
    def export(self) -> Path:
        """Convert the model with the target's toolchain.

        Returns:
            Path of the converted model, still under its intermediate
            name.

        """

    def run(self) -> Path:
        """Convert the model and finish the output directory.

        Runs `export`, renames its result after the original model,
        drops the intermediate outputs unless they are to be kept, and
        writes ``buildinfo.json``.

        Returns:
            Path of the converted model in the output directory.

        """
        output_path = self.export()
        new_output_path = self.output_dir / Path(
            self.original_model_name
        ).with_suffix(output_path.suffix)
        shutil.move(
            str(output_path),
            new_output_path,
        )
        if self._inference_model_path == output_path:
            self._inference_model_path = new_output_path

        if not self.keep_intermediate_outputs:  # pragma: no cover
            shutil.rmtree(self.intermediate_outputs_dir)

        buildinfo = {
            "cmd_info": self._cmd_info,
            "modelconverter_version": version("modelconv"),
            **self.exporter_buildinfo(),
        }

        with open(self.output_dir / "buildinfo.json", "w") as f:
            json.dump(buildinfo, f, indent=4)

        return new_output_path

    def read_img_dir(self, path: Path, max_images: int) -> list[Path]:
        """Collect the calibration images from a directory.

        If the directory holds no images, the error is logged and the
        process exits.

        Args:
            path: Directory to read the calibration images from.
            max_images: How many of the images to use. A negative value
                means all of them.

        Returns:
            Paths of the images to calibrate with, sorted by file name.

        """
        imgs = read_calib_dir(path)
        if not imgs:
            exit_with(FileNotFoundError(f"No images found in {path}"))
        imgs = sorted(imgs, key=lambda x: x.name)
        if max_images >= 0:
            logger.info(
                f"Using [{max_images}/{len(imgs)}] images for calibration."
            )
            imgs = imgs[:max_images]
        return imgs

    def _prepare_random_calibration_data(self) -> None:
        for name, inp in self.inputs.items():
            calib = inp.calibration
            if not isinstance(calib, RandomCalibrationConfig):
                continue
            logger.warning(
                f"Random calibration is being used for input '{name}'."
            )
            dest = self.intermediate_outputs_dir / "random" / name
            dest.mkdir(parents=True)
            if inp.shape is None:  # pragma: no cover
                exit_with(
                    ValueError(
                        f"Random calibration requires shape to be specified for input '{name}'."
                    )
                )

            for i in range(calib.max_images):
                arr = np.random.normal(calib.mean, calib.std, inp.shape)
                arr = np.clip(arr, calib.min_value, calib.max_value)

                arr = arr.astype(calib.data_type.as_numpy_dtype())
                if not inp.is_raw_input and (
                    len(arr.shape) in {2, 3}
                    or (len(arr.shape) in {3, 4} and arr.shape[0] == 1)
                ):
                    layout = inp.layout
                    if arr.shape[0] == 1 and len(arr.shape) > 2:
                        arr = arr.squeeze(0)
                        if layout is not None:
                            layout = layout[1:]

                    if layout is not None and "C" in layout:
                        channel_dim = layout.index("C")
                        if channel_dim == 0 and len(arr.shape) == 3:
                            arr = arr.transpose(1, 2, 0)
                    elif arr.shape[0] in {1, 3}:  # type: ignore
                        arr = arr.transpose(1, 2, 0)
                    cv2.imwrite(str(dest / f"{i}.png"), arr)
                else:
                    np.save(dest / f"{i}.npy", arr)

            self.inputs[name].calibration = ImageCalibrationConfig(path=dest)

    @staticmethod
    def _attach_suffix(path: Path | str, suffix: str) -> Path:
        return Path(str(Path(path).with_suffix("")) + f"-{suffix.lstrip('-')}")

    @staticmethod
    def _add_args(args: list, new_args: list, index: int = 0) -> None:
        if new_args[index] not in args:
            args.extend(new_args)

    def _subprocess_run(
        self, args: list[str], meta_name: str, **kwargs
    ) -> SubprocessResult:
        result = subprocess_run(args, **kwargs)
        self._cmd_info[meta_name] = [str(arg) for arg in args]
        return result
