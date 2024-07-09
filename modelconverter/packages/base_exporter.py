import json
import shutil
from abc import ABC, abstractmethod
from importlib.metadata import version
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import onnx

from modelconverter.utils import read_calib_dir, subprocess_run
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    RandomCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.exceptions import exit_with
from modelconverter.utils.types import InputFileType, Target

logger = getLogger(__name__)


class Exporter(ABC):
    target: Target

    def __init__(
        self,
        config: SingleStageConfig,
        output_dir: Path,
    ):
        self.config = config
        self.output_dir = output_dir
        self.input_model = config.input_model
        self.input_file_type = config.input_file_type
        self.inputs = {inp.name: inp for inp in config.inputs}
        self._inference_model_path: Optional[Path] = None

        self.outputs = {out.name: out for out in config.outputs}
        self.keep_intermediate_outputs = config.keep_intermediate_outputs
        self.disable_onnx_simplification = config.disable_onnx_simplification

        self.model_name = self.input_model.stem

        self.intermediate_outputs_dir = (
            self.output_dir / "intermediate_outputs"
        )
        self.intermediate_outputs_dir.mkdir(parents=True, exist_ok=True)

        self._cmd_info: Dict[str, List[str]] = {}
        self.is_tflite = self.input_file_type == InputFileType.TFLITE

        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(config.model_dump_json(indent=4))

        shutil.copy(self.input_model, self.intermediate_outputs_dir)
        if self.input_file_type == InputFileType.IR:
            assert self.config.input_bin is not None
            shutil.copy(self.config.input_bin, self.intermediate_outputs_dir)
        shutil.copy(self.input_model, self.output_dir)
        self.input_model = (
            self.intermediate_outputs_dir / self.input_model.name
        )

        if (
            not self.disable_onnx_simplification
            and self.input_file_type == InputFileType.ONNX
        ):
            self.input_model = self.simplify_onnx()

        self._disable_calibration = getattr(
            self.config, self.target.name.lower()
        ).disable_calibration

        if self._disable_calibration:
            logger.warning("Calibration has been disabled.")
            logger.warning("The quantization step will be skipped.")

        for name, inp in self.inputs.items():
            calib = inp.calibration
            if calib is None:
                continue
            if not isinstance(calib, RandomCalibrationConfig):
                continue
            logger.warning(
                f"Random calibration is being used for input '{name}'."
            )
            shape = cast(List[int], inp.shape)
            dest = self.intermediate_outputs_dir / "random" / name
            dest.mkdir(parents=True)
            if shape is None or not all(isinstance(dim, int) for dim in shape):
                exit_with(
                    ValueError(
                        f"Random calibration requires shape to be specified for input '{name}'."
                    )
                )

            for i in range(calib.max_images):
                arr = np.random.normal(calib.mean, calib.std, shape)
                arr = np.clip(arr, calib.min_value, calib.max_value)

                arr = arr.astype(calib.data_type.as_numpy_dtype())
                np.save(dest / f"{i}.npy", arr)

            self.inputs[name].calibration = ImageCalibrationConfig(path=dest)

    @property
    def inference_model_path(self) -> Path:
        if self._inference_model_path is None:
            raise ValueError(
                "Inference model path not yet set. Export must be run first."
            )
        return self._inference_model_path

    def simplify_onnx(self) -> Path:
        logger.info("Simplifying ONNX.")
        try:
            from onnxsim import simplify
        except ImportError:
            logger.warning(
                "onnxsim not installed, proceeding without simplification."
                "Please install it using `pip install onnxsim`."
            )
            return self.input_model

        onnx_sim, check = simplify(str(self.input_model))
        if not check:
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
        onnx.save(onnx_sim, str(onnx_sim_path))
        return onnx_sim_path

    @abstractmethod
    def exporter_buildinfo(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def export(self) -> Path:
        pass

    def run(self) -> Path:
        output_path = self.export()
        new_output_path = (
            self.output_dir
            / Path(self.model_name).with_suffix(output_path.suffix).name
        )
        shutil.move(
            str(output_path),
            new_output_path,
        )
        if not self.keep_intermediate_outputs:
            shutil.rmtree(self.intermediate_outputs_dir)

        buildinfo = {
            "cmd_info": self._cmd_info,
            "modelconverter_version": version("modelconv"),
            **self.exporter_buildinfo(),
        }

        with open(self.output_dir / "buildinfo.json", "w") as f:
            json.dump(buildinfo, f, indent=4)

        return new_output_path

    def read_img_dir(self, path: Path, max_images: int) -> List[Path]:
        imgs = read_calib_dir(path)
        if not imgs:
            exit_with(FileNotFoundError(f"No images found in {path}"))
        if max_images >= 0:
            logger.info(
                f"Using [{max_images}/{len(imgs)}] images for calibration."
            )
            imgs = imgs[:max_images]
        return imgs

    @staticmethod
    def _attach_suffix(path: Union[Path, str], suffix: str) -> Path:
        return Path(str(Path(path).with_suffix("")) + f"-{suffix.lstrip('-')}")

    @staticmethod
    def _add_args(args: list, new_args: list, index=0):
        if new_args[index] not in args:
            args.extend(new_args)

    def _subprocess_run(self, args: List[str], meta_name: str, **kwargs):
        subprocess_run(args, **kwargs)
        self._cmd_info[meta_name] = [str(arg) for arg in args]
