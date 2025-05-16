import json
from pathlib import Path

import cv2
from loguru import logger

from modelconverter.packages.base_exporter import Exporter
from modelconverter.packages.rvc2.exporter import RVC2Exporter
from modelconverter.utils import exit_with, read_image
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.subprocess import subprocess_run
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    Target,
)


class RVC3Exporter(RVC2Exporter):
    target: Target = Target.RVC3

    def __init__(self, config: SingleStageConfig, output_dir: Path):
        Exporter.__init__(self, config=config, output_dir=output_dir)

        self.compress_to_fp16 = config.rvc3.compress_to_fp16
        self.pot_target_device = config.rvc3.pot_target_device
        self.mo_args = config.rvc3.mo_args
        self.compile_tool_args = config.rvc3.compile_tool_args
        self.device = "VPUX.3400"
        self._device_specific_buildinfo = {}

    def export(self) -> Path:
        if self.input_file_type == InputFileType.TFLITE:
            self._transform_tflite_to_onnx()

        if self.input_file_type == InputFileType.ONNX:
            xml_path = self._export_openvino_ir()
        elif self.input_file_type == InputFileType.IR:
            xml_path = self.input_model
        else:
            raise NotImplementedError

        self._inference_model_path = xml_path
        args = self.compile_tool_args
        self._add_args(args, ["-d", self.device])
        self._add_args(args, ["-ip", "U8"])

        if not self._disable_calibration:
            if len(self.inputs) > 1:
                exit_with(
                    NotImplementedError(
                        "Quantization is not yet supported for"
                        "models with multiple inputs."
                    )
                )
            calibrated_xml_path = self.calibrate(xml_path)
            self._inference_model_path = calibrated_xml_path
            output_path = (
                self.output_dir
                / f"{self.model_name}-{self.target.name.lower()}-int8"
            )
            args += ["-m", calibrated_xml_path]
        else:
            output_path = (
                self.output_dir
                / f"{self.model_name}-{self.target.name.lower()}"
            )
            args += ["-m", xml_path]

        if "-o" not in args:
            blob_output_path = output_path.with_suffix(".blob")
            args += ["-o", blob_output_path]
        else:
            blob_output_path = Path(args[args.index("-o") + 1])

        self._subprocess_run(["compile_tool", *args], meta_name="compile_tool")
        logger.info(f"OpenVINO IR compiled to {self.output_dir}")
        return blob_output_path

    def calibrate(self, xml_path: Path) -> Path:
        inp = next(iter(self.inputs.values()))
        calib = inp.calibration
        assert isinstance(calib, ImageCalibrationConfig)

        files = self.read_img_dir(calib.path, calib.max_images)
        calibration_img_dir = (
            self.intermediate_outputs_dir / "calibration_images"
        )
        calibration_img_dir.mkdir(exist_ok=True)

        for file in files:
            if inp.shape is None:
                raise ValueError(
                    "Input shape must be provided for calibration"
                )
            img = read_image(
                file,
                inp.shape,
                inp.encoding.to,
                calib.resize_method,
                data_type=DataType.UINT8,
                transpose=False,
            )
            cv2.imwrite(str(calibration_img_dir / file.name), img)

        config = {
            "model": {
                "model_name": f"{xml_path.stem}-int8",
                "model": str(xml_path),
                "weights": str(xml_path.with_suffix(".bin")),
            },
            "engine": {
                "launchers": [
                    {
                        "framework": "openvino",
                        "device": "CPU",
                    }
                ],
                "datasets": [
                    {
                        "name": "calibration",
                        "data_source": str(calibration_img_dir),
                        "reader": "opencv_imread",
                    }
                ],
            },
            "compression": {
                "target_device": self.pot_target_device.name,
                "algorithms": [
                    {
                        "name": "DefaultQuantization",
                        "params": {
                            "preset": "performance",
                            "stat_subset_size": 300,
                        },
                    }
                ],
            },
        }

        if inp.encoding.to == Encoding.GRAY:
            config["engine"]["datasets"][0]["preprocessing"] = [
                {"type": "bgr_to_gray"}
            ]
        elif not self.reverse_input_channels:
            config["engine"]["datasets"][0]["preprocessing"] = [
                {"type": "bgr_to_rgb"}
            ]

        pot_config_path = self.intermediate_outputs_dir / "pot_config.json"

        with open(pot_config_path, "w") as f:
            json.dump(config, f, indent=4)

        logger.info(f"Executing POT pipeline for {xml_path}")

        subprocess_run(
            [
                "pot",
                "--config",
                pot_config_path,
                "-d",
                "--output-dir",
                self.intermediate_outputs_dir,
            ],
        )

        logger.info("Calibration finished successfully")
        return Path(
            self.intermediate_outputs_dir
            / "optimized"
            / f"{xml_path.stem}-int8.xml"
        )
