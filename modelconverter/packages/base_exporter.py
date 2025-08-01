import json
import shutil
from abc import ABC, abstractmethod
from importlib.metadata import version
from pathlib import Path
from typing import Any

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
from modelconverter.utils.types import InputFileType, Target


class Exporter(ABC):
    target: Target

    def __init__(
        self,
        config: SingleStageConfig,
        output_dir: Path,
    ):
        input_model = config.input_model

        self.config = config
        self.output_dir = output_dir
        self.input_file_type = config.input_file_type
        self.inputs = {inp.name: inp for inp in config.inputs}
        self._inference_model_path: Path | None = None

        self.outputs = {out.name: out for out in config.outputs}
        self.keep_intermediate_outputs = config.keep_intermediate_outputs
        self.disable_onnx_simplification = config.disable_onnx_simplification
        self.disable_onnx_optimization = config.disable_onnx_optimization

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
        if input_model.with_suffix(".onnx_data").exists():
            shutil.copy(
                input_model.with_suffix(".onnx_data"),
                self.intermediate_outputs_dir,
            )
            shutil.copy(
                input_model.with_suffix(".onnx_data"),
                self.output_dir,
            )
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
            not self.disable_onnx_simplification
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
        if self.input_model.with_suffix(".onnx_data").exists():
            onnx.save(
                onnx_sim,
                str(onnx_sim_path),
                save_as_external_data=True,
                location=f"{onnx_sim_path.name}_data",
            )
        else:
            onnx.save(onnx_sim, str(onnx_sim_path))
        return onnx_sim_path

    @abstractmethod
    def exporter_buildinfo(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def export(self) -> Path:
        pass

    def run(self) -> Path:
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

    def read_img_dir(self, path: Path, max_images: int) -> list[Path]:
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
            if inp.shape is None:
                exit_with(
                    ValueError(
                        f"Random calibration requires shape to be specified for input '{name}'."
                    )
                )

            for i in range(calib.max_images):
                arr = np.random.normal(calib.mean, calib.std, inp.shape)
                arr = np.clip(arr, calib.min_value, calib.max_value)

                arr = arr.astype(calib.data_type.as_numpy_dtype())
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
    ) -> None:
        subprocess_run(args, **kwargs)
        self._cmd_info[meta_name] = [str(arg) for arg in args]
