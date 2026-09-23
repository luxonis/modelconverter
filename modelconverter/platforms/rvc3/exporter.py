"""Export of models to the RVC3 platform.

The pipeline is the RVC2 one -- ``mo`` produces an OpenVINO IR that
``compile_tool`` compiles into a blob -- with INT8 post-training
quantization by OpenVINO's POT inserted before the compilation. It runs
inside the RVC3 Docker image.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from luxonis_ml.typing import Params

from modelconverter.platforms.base_exporter import Exporter
from modelconverter.platforms.rvc2.exporter import RVC2Exporter
from modelconverter.utils import ModelconverterException, exit_with, read_image
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    InputConfig,
    SingleStageConfig,
)
from modelconverter.utils.preprocessing import (
    channels_last_4d_layout,
    is_user_calibration_tensor,
    read_user_calibration_tensor,
    reorder_layout,
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
            self._validate_ir_preprocessing_contract()
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

        The calibration data configured for the model's single input is
        written into the intermediate outputs directory -- as NumPy
        samples when it holds values no image can carry, as images
        otherwise. A POT config pointing at it is generated, and ``pot``
        is run on that. Requires the input to use file calibration data.

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
        if inp.shape is None:  # pragma: no cover
            raise ValueError("Input shape must be provided for calibration")

        # Normalized data cannot safely be written back to an image. POT's
        # NumPy reader preserves floating-point and negative values and also
        # lets `.npy`/`.raw` keep their model-ready pass-through contract.
        if (
            inp.calibration_preprocessing is not None
            or calib.generated_from_random
            or any(file.suffix.lower() in {".npy", ".raw"} for file in files)
        ):
            dataset = self._write_calibration_tensors(
                inp, calib, files, shape=inp.shape
            )
        else:
            dataset = self._write_calibration_images(
                inp, calib, files, shape=inp.shape
            )

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

    def _write_calibration_tensors(
        self,
        inp: InputConfig,
        calib: ImageCalibrationConfig,
        files: list[Path],
        *,
        shape: list[int],
    ) -> Params:
        """Write one NumPy sample per calibration file, for POT's numpy_reader.

        Args:
            inp: Input the calibration data belongs to.
            calib: Image calibration configuration of that input.
            files: Calibration files to convert.
            shape: Shape of the model input, already resolved.

        Returns:
            The POT dataset description pointing at the written samples.

        Raises:
            ValueError: If the input has no layout.
            ModelconverterException: If a sample does not have the shape the
                backend input asks for.

        """
        if inp.layout is None:  # pragma: no cover - shape resolves layout
            raise ValueError("Input layout must be provided for calibration")

        # Accuracy Checker's NumPy reader treats each file as one sample. Its
        # input feeder adds the batch axis and, for a four-dimensional
        # OpenVINO input, converts an HWC sample to the model layout.
        # Persisting the full NCHW tensor here would therefore make POT
        # interpret its axes as an NHWC sample.
        sample_layout = channels_last_4d_layout(inp.layout).replace("N", "", 1)
        expected_shape = tuple(
            shape[inp.layout.index(axis)] for axis in sample_layout
        )

        directory = self.intermediate_outputs_dir / "calibration_tensors"
        directory.mkdir(exist_ok=True)
        for index, file in enumerate(files):
            if is_user_calibration_tensor(file, calib):
                array = read_user_calibration_tensor(
                    file,
                    raw_shape=expected_shape,
                    data_type=inp.data_type,
                    input_name=inp.name,
                )
            else:
                array, layout = self._read_calibration_file(inp, calib, file)
                array = reorder_layout(array, layout, sample_layout)
            if array.shape != expected_shape:
                raise ModelconverterException(
                    f"Calibration data for input '{inp.name}' has shape "
                    f"{list(array.shape)}, expected {list(expected_shape)}."
                )
            np.save(directory / f"{index}.npy", array)

        return {
            "name": "calibration",
            "data_source": str(directory),
            "reader": "numpy_reader",
        }

    def _write_calibration_images(
        self,
        inp: InputConfig,
        calib: ImageCalibrationConfig,
        files: list[Path],
        *,
        shape: list[int],
    ) -> Params:
        """Write the calibration images resized to the input, for POT.

        Args:
            inp: Input the calibration data belongs to.
            calib: Image calibration configuration of that input.
            files: Calibration files to convert.
            shape: Shape of the model input, already resolved.

        Returns:
            The POT dataset description pointing at the written images,
            including grayscale conversion when POT has to apply it.

        """
        directory = self.intermediate_outputs_dir / "calibration_images"
        directory.mkdir(exist_ok=True)
        for file in files:
            img = read_image(
                file,
                shape,
                inp.encoding.to,
                calib.resize_method,
                data_type=DataType.UINT8,
                transpose=False,
            )
            cv2.imwrite(str(directory / file.name), img)

        dataset: Params = {
            "name": "calibration",
            "data_source": str(directory),
            "reader": "opencv_imread",
        }
        if inp.encoding.to == Encoding.GRAY:
            dataset["preprocessing"] = [{"type": "bgr_to_gray"}]
        return dataset
