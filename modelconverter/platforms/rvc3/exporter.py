"""Export of models to the RVC3 platform.

The pipeline is the RVC2 one -- ``mo`` produces an OpenVINO IR that
``compile_tool`` compiles into a blob -- with INT8 post-training
quantization by OpenVINO's POT inserted before the compilation. It runs
inside the RVC3 Docker image.
"""

import json
from pathlib import Path

import cv2
from loguru import logger
from luxonis_ml.typing import Params

from modelconverter.platforms.base_exporter import Exporter
from modelconverter.platforms.rvc2.exporter import RVC2Exporter
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
    Platform,
)


class RVC3Exporter(RVC2Exporter):
    """Exporter producing an RVC3 blob, quantized to INT8 by default.

    Reuses the OpenVINO conversion of `RVC2Exporter` but compiles for
    the ``VPUX.3400`` device and, unless calibration is disabled,
    quantizes the IR with POT before compiling it.
    """

    platform: Platform = Platform.RVC3

    def __init__(self, config: SingleStageConfig, output_dir: Path):
        """Initialize the exporter from the RVC3 configuration.

        Args:
            config: Configuration of the stage to export. Its ``rvc3``
                section supplies the RVC3-specific options.
            output_dir: Directory the compiled model and the build
                information are written to.

        """
        Exporter.__init__(self, config=config, output_dir=output_dir)

        self._compress_to_fp16 = config.rvc3.compress_to_fp16
        self._pot_target_device = config.rvc3.pot_target_device
        self._mo_args = config.rvc3.mo_args
        self._compile_tool_args = config.rvc3.compile_tool_args
        self._device = "VPUX.3400"
        self._reverse_input_channels = False
        self._device_specific_buildinfo = {}

    def export(self) -> Path:
        """Convert the model and compile it for RVC3.

        A TFLite input is first converted to ONNX, an ONNX input is
        converted to an OpenVINO IR and an IR input is used as it is.
        Unless calibration is disabled, the IR is then quantized to
        INT8 and the quantized model is the one compiled; in that case
        the process exits with an error for models with more than one
        input, which quantization does not support yet.

        Returns:
            Path to the compiled ``.blob``.

        Raises:
            NotImplementedError: If the input file type is neither
                TFLite, ONNX nor OpenVINO IR.

        """
        if self._input_file_type == InputFileType.TFLITE:
            self._transform_tflite_to_onnx()

        if self._input_file_type == InputFileType.ONNX:
            xml_path = self._export_openvino_ir()
        elif self._input_file_type == InputFileType.IR:
            xml_path = self._input_model
        else:
            raise NotImplementedError

        self._inference_model_path = xml_path
        args = self._compile_tool_args
        self._add_args(args, ["-d", self._device])
        if "-iop" not in args:
            self._add_args(args, ["-ip", "U8"])

        if not self._disable_calibration:
            if len(self._inputs) > 1:
                exit_with(
                    NotImplementedError(
                        "Quantization is not yet supported for"
                        "models with multiple inputs."
                    )
                )
            calibrated_xml_path = self._calibrate(xml_path)
            self._inference_model_path = calibrated_xml_path
            output_path = (
                self.output_dir
                / f"{self._model_name}-{self.platform.name.lower()}-int8"
            )
            args += ["-m", str(calibrated_xml_path)]
        else:
            output_path = (
                self.output_dir
                / f"{self._model_name}-{self.platform.name.lower()}"
            )
            args += ["-m", str(xml_path)]

        if "-o" not in args:
            blob_output_path = output_path.with_suffix(".blob")
            args += ["-o", str(blob_output_path)]
        else:  # pragma: no cover
            blob_output_path = Path(args[args.index("-o") + 1])

        self._subprocess_run(["compile_tool", *args], meta_name="compile_tool")
        logger.info(f"OpenVINO IR compiled to {self.output_dir}")
        return blob_output_path

    def _calibrate(self, xml_path: Path) -> Path:
        """Quantize an OpenVINO IR to INT8 with the POT tool.

        The calibration images configured for the model's single input
        are read, resized to the input shape and written into the
        intermediate outputs directory, a POT config pointing at them
        is generated, and ``pot`` is run on it. Requires the input to
        use image calibration data.

        Args:
            xml_path: Path to the ``.xml`` of the IR to quantize.

        Returns:
            Path to the ``.xml`` of the quantized IR.

        Raises:
            ValueError: If the input has no shape.

        """
        inp = next(iter(self._inputs.values()))
        calib = inp.calibration
        assert isinstance(calib, ImageCalibrationConfig)

        files = self._read_img_dir(calib.path, calib.max_images)
        calibration_img_dir = (
            self.intermediate_outputs_dir / "calibration_images"
        )
        calibration_img_dir.mkdir(exist_ok=True)

        for file in files:
            if inp.shape is None:  # pragma: no cover
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
            suffix = ".png" if file.suffix == ".npy" else file.suffix
            cv2.imwrite(
                str((calibration_img_dir / file.stem).with_suffix(suffix)), img
            )

        dataset: Params = {
            "name": "calibration",
            "data_source": str(calibration_img_dir),
            "reader": "opencv_imread",
        }
        if inp.encoding.to == Encoding.GRAY:
            dataset["preprocessing"] = [{"type": "bgr_to_gray"}]
        elif not self._reverse_input_channels:
            dataset["preprocessing"] = [{"type": "bgr_to_rgb"}]

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
                "datasets": [dataset],
            },
            "compression": {
                "target_device": self._pot_target_device.name,
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
