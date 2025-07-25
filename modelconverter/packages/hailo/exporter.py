import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hailo_sdk_client
import numpy as np
import tensorflow as tf
from hailo_sdk_client import ClientRunner
from loguru import logger

from modelconverter.packages.base_exporter import Exporter
from modelconverter.utils import exit_with, read_image
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.types import Target


class HailoExporter(Exporter):
    target: Target = Target.HAILO

    def __init__(self, config: SingleStageConfig, output_dir: Path):
        super().__init__(config=config, output_dir=output_dir)
        self.optimization_level = config.hailo.optimization_level
        self.compression_level = config.hailo.compression_level
        self.batch_size = config.hailo.batch_size
        self.disable_compilation = config.hailo.disable_compilation
        self._alls: list[str] = []
        self.hw_arch = config.hailo.hw_arch
        if not tf.config.list_physical_devices("GPU"):
            logger.error(
                "No GPU found. Setting optimization and compression level to 0."
            )
            self.optimization_level = 0
            self.compression_level = 0

    def _get_start_nodes(self) -> tuple[list[str], dict[str, list[int]]]:
        start_nodes = []
        net_input_shapes = {}
        for name, inp in self.inputs.items():
            start_nodes.append(name)
            if inp.shape is not None:
                net_input_shapes[inp.name] = inp.shape
        return start_nodes, net_input_shapes

    def export(self) -> Path:
        runner = ClientRunner(hw_arch=self.hw_arch)
        start_nodes, net_input_shapes = self._get_start_nodes()

        logger.info("Translating model to Hailo IR.")
        if self.is_tflite:
            cast(Callable[..., None], runner.translate_tf_model)(
                str(self.input_model),
                net_name=self.model_name,
                start_node_names=start_nodes,
                tensor_shapes=net_input_shapes,
                end_node_names=list(self.outputs.keys()),
            )
        else:
            cast(Callable[..., None], runner.translate_onnx_model)(
                str(self.input_model),
                net_name=self.model_name,
                start_node_names=start_nodes,
                net_input_shapes=net_input_shapes,
                end_node_names=list(self.outputs.keys()),
            )
        logger.info("Model translated to Hailo IR.")
        har_path = self.input_model.with_suffix(".har")
        runner.save_har(har_path)
        if self._disable_calibration:
            self._inference_model_path = self.output_dir / Path(
                self.original_model_name
            ).with_suffix(".har")
            return har_path

        quantized_har_path = self._calibrate(har_path)
        self._inference_model_path = Path(quantized_har_path)
        if self.disable_compilation:
            logger.warning("Compilation disabled, skipping compilation.")
            copy_path = Path(quantized_har_path).parent / (
                Path(quantized_har_path).stem + "_copy.har"
            )
            shutil.copy(
                quantized_har_path,
                copy_path,
            )
            return copy_path

        runner = ClientRunner(hw_arch=self.hw_arch, har=quantized_har_path)
        hef = runner.compile()

        hef_path = self.input_model.with_suffix(".hef")
        with open(hef_path, "wb") as hef_file:
            hef_file.write(hef)
        return hef_path

    def _get_calibration_data(
        self, runner: ClientRunner
    ) -> dict[str, np.ndarray]:
        data = {}
        for orig_name, inp in self.inputs.items():
            name, shape = self._get_hn_layer_info(runner, orig_name)
            shape = shape[1:]

            calib = inp.calibration
            assert isinstance(calib, ImageCalibrationConfig)

            images = self.read_img_dir(calib.path, calib.max_images)
            calib_dataset = np.zeros((len(images), *shape), dtype=np.float32)

            if len(shape) == 3:
                H, W, C = shape
                shape = [C, H, W]

            for idx, img_path in enumerate(images):
                img = read_image(
                    img_path,
                    [1, *shape],
                    inp.encoding.to,
                    calib.resize_method,
                    data_type=inp.data_type,
                    transpose=False,
                )
                if len(shape) == 3 and img.shape == (1, *shape):
                    img = np.transpose(img, (0, 2, 3, 1))

                calib_dataset[idx] = img

            data[name] = calib_dataset

        return data

    def _calibrate(self, har_path: Path) -> str:
        logger.info("Calibrating model.")

        runner = ClientRunner(hw_arch=self.hw_arch, har=str(har_path))
        alls = self._get_alls(runner)
        logger.info(f"Using the following configuration: {alls}")

        calib_dataset = self._get_calibration_data(runner)

        runner.load_model_script(alls)
        runner.optimize(calib_dataset)

        quantized_model_har_path = self._attach_suffix(
            har_path, "quantized.har"
        )
        runner.save_har(quantized_model_har_path)
        logger.info("Model calibration finished.")
        return str(quantized_model_har_path)

    @staticmethod
    def _get_hn_layer_info(
        runner: ClientRunner, name: str
    ) -> tuple[str, list[int]]:
        for hn_name, params in runner.get_hn_dict()["layers"].items():
            if name in params.get("original_names", []):
                return hn_name, [1, *(params["input_shapes"][0])[1:]]
        raise RuntimeError(
            f"Could not find HN layer name for {name}. This should not happen."
        )

    def _get_alls(self, runner: ClientRunner) -> str:
        alls = self.config.hailo.alls
        alls.append(
            f"model_optimization_flavor("
            f"optimization_level={self.optimization_level}, "
            f"compression_level={self.compression_level}, "
            f"batch_size={self.batch_size})"
        )
        for name, inp in self.inputs.items():
            safe_name = name.replace(".", "")

            hn_name, _ = self._get_hn_layer_info(runner, name)

            if inp.shape is None:
                exit_with(
                    ValueError(f"Input `{name}` has no shape specified.")
                )
            if not all(x is not None for x in inp.shape):
                exit_with(ValueError(f"Input `{name}` has dynamic shape."))

            if self.is_tflite:
                values_len = inp.shape[-1]
            else:
                values_len = inp.shape[1]

            assert values_len is not None
            scale_values = inp.scale_values or [1.0] * values_len
            mean_values = inp.mean_values or [0.0] * values_len
            alls.append(
                f"normalization_{safe_name} = normalization("
                f"{mean_values},{scale_values},{hn_name})"
            )

            if inp.encoding_mismatch:
                alls.append(
                    f"bgr_to_rgb_{safe_name} = input_conversion("
                    f"{hn_name},bgr_to_rgb)"
                )

        self._alls = alls
        return "\n".join(alls)

    def exporter_buildinfo(self) -> dict[str, Any]:
        return {
            "hailo_version": hailo_sdk_client.__version__,
            "optimization_level": self.optimization_level,
            "compression_level": self.compression_level,
            "alls": self._alls,
        }
