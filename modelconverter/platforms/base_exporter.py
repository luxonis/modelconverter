import json
import shutil
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import onnx
from loguru import logger
from luxonis_ml.typing import Params, PathType

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
from modelconverter.utils.types import InputFileType, Platform


class Exporter(ABC):
    platform: Platform

    def __init__(
        self,
        config: SingleStageConfig,
        output_dir: Path,
    ):
        input_model = config.input_model

        self.config = config
        self.output_dir = output_dir
        self._input_file_type = config.input_file_type
        self._inputs = {inp.name: inp for inp in config.inputs}
        self._inference_model_path: Path | None = None

        self._outputs = {out.name: out for out in config.outputs}
        self._keep_intermediate_outputs = config.keep_intermediate_outputs
        self._onnx_simplification = config.onnx_simplification
        self._onnx_optimizations = config.onnx_optimizations

        self._model_name = sanitize_net_name(input_model.stem)
        self._original_model_name = sanitize_net_name(
            input_model.name, with_suffix=True
        )

        self.intermediate_outputs_dir = (
            self.output_dir / "intermediate_outputs"
        )
        self.intermediate_outputs_dir.mkdir(parents=True, exist_ok=True)

        self._cmd_info: dict[str, list[str]] = {}
        self._is_tflite = self._input_file_type == InputFileType.TFLITE

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
            if self._input_file_type == InputFileType.ONNX
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
        if self._input_file_type == InputFileType.IR:
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
        self._input_model = (
            self.intermediate_outputs_dir / sanitized_model_name
        )

        if (
            self._onnx_simplification
            and self._input_file_type == InputFileType.ONNX
        ):
            self._input_model = self._simplify_onnx()

        self._disable_calibration = getattr(
            self.config, self.platform.name.lower()
        ).disable_calibration

        if self.platform != Platform.RVC2 and self._disable_calibration:
            logger.warning("Calibration has been disabled.")
            logger.warning("The quantization step will be skipped.")

        if self.platform != Platform.RVC2 and not self._disable_calibration:
            self._prepare_random_calibration_data()

    @property
    def inference_model_path(self) -> Path:
        if self._inference_model_path is None:  # pragma: no cover
            raise ValueError(
                "Inference model path not yet set. Export must be run first."
            )
        return self._inference_model_path

    def _simplify_onnx(self) -> Path:  # pragma: no cover
        logger.info("Simplifying ONNX.")
        try:
            if self._onnx_simplification == "onnxsim":
                from onnxsim import simplify

                logger.info("Using `onnxsim` for simplification.")
            elif self._onnx_simplification == "onnxslim":
                from onnxslim import slim

                logger.info("Using `onnxslim` for simplification.")

                def simplify(model: str) -> tuple[onnx.ModelProto, bool]:
                    slimmed = slim(onnx.load(model))
                    assert isinstance(slimmed, onnx.ModelProto)
                    return slimmed, True

        except ImportError:
            backend = self._onnx_simplification
            logger.warning(
                f"`{backend}` not installed, proceeding without simplification."
                f"Please install it using `pip install {backend}`."
            )
            return self._input_model

        try:
            onnx_sim, check = simplify(str(self._input_model))
        except Exception as e:  # pragma: no cover
            logger.warning(
                f"Failed to simplify ONNX: {e}. Proceeding without simplification."
            )
            return self._input_model
        if not check:  # pragma: no cover
            logger.warning(
                "Provided ONNX could not be simplified. "
                "Proceeding without simplification."
            )
            return self._input_model
        logger.info("ONNX successfully simplified.")
        onnx_sim_path = self._attach_suffix(
            self._input_model, "simplified.onnx"
        )
        logger.info(f"Saving simplified ONNX to {onnx_sim_path}")
        save_onnx_model(
            onnx_sim,
            onnx_sim_path,
            save_as_external_data=has_external_data(self._input_model),
            location=f"{onnx_sim_path.name}_data",
        )
        return onnx_sim_path

    @abstractmethod
    def exporter_buildinfo(self) -> Params:
        pass

    @abstractmethod
    def export(self) -> Path:
        pass

    def run(self) -> Path:
        output_path = self.export()
        new_output_path = self.output_dir / Path(
            self._original_model_name
        ).with_suffix(output_path.suffix)
        shutil.move(
            str(output_path),
            new_output_path,
        )
        if self._inference_model_path == output_path:
            self._inference_model_path = new_output_path

        if not self._keep_intermediate_outputs:  # pragma: no cover
            shutil.rmtree(self.intermediate_outputs_dir)

        buildinfo = {
            "cmd_info": self._cmd_info,
            "modelconverter_version": version("modelconv"),
            **self.exporter_buildinfo(),
        }

        with open(self.output_dir / "buildinfo.json", "w") as f:
            json.dump(buildinfo, f, indent=4)

        return new_output_path

    def _read_img_dir(self, path: Path, max_images: int) -> list[Path]:
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
        for name, inp in self._inputs.items():
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
                arr: np.ndarray = np.random.normal(
                    calib.mean, calib.std, inp.shape
                )
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
                    elif len(arr.shape) == 3 and arr.shape[0] in {1, 3}:
                        arr = arr.transpose(1, 2, 0)
                    cv2.imwrite(str(dest / f"{i}.png"), arr)
                else:
                    np.save(dest / f"{i}.npy", arr)

            self._inputs[name].calibration = ImageCalibrationConfig(path=dest)

    @staticmethod
    def _attach_suffix(path: PathType, suffix: str) -> Path:
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
