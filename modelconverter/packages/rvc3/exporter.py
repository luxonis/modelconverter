from logging import getLogger
from pathlib import Path

import addict
import numpy as np
import openvino.tools.pot as pot
from luxonis_ml.utils.logging import reset_logging, setup_logging

from modelconverter.utils import exit_with, read_image
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.types import InputFileType, Target

from ..base_exporter import Exporter
from ..rvc2.exporter import COMPILE_TOOL, RVC2Exporter

# NOTE: importing `pot` breaks the logging module, so we need to reset it
reset_logging()
setup_logging(file="modelconverter.log", use_rich=True)

logger = getLogger(__name__)


class RVC3Exporter(RVC2Exporter):
    target: Target = Target.RVC3

    def __init__(self, config: SingleStageConfig, output_dir: Path):
        Exporter.__init__(self, config=config, output_dir=output_dir)

        self.pot_target_device = config.rvc3.pot_target_device
        self.mo_args = config.rvc3.mo_args
        self.compile_tool_args = config.rvc3.compile_tool_args
        self.device = "VPUX.3400"
        self._device_specific_buildinfo = {}

    def export(self) -> Path:
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

        output_dir = Path(self.intermediate_outputs_dir)

        if not self._disable_calibration:
            if len(self.inputs) > 1:
                exit_with(
                    NotImplementedError(
                        "Quantization is not yet supported for"
                        "models with multiple inputs."
                    )
                )
            calibrated_xml_path = self.calibrate(
                xml_path,
                output_dir=str(output_dir),
            )
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

        self._subprocess_run([COMPILE_TOOL, *args], meta_name="compile_tool")
        logger.info(f"OpenVINO IR compiled to {self.output_dir}")
        return blob_output_path

    def calibrate(self, xml_path: Path, output_dir: str) -> Path:
        inp = list(self.inputs.values())[0]
        calib = inp.calibration
        assert isinstance(calib, ImageCalibrationConfig)

        files = self.read_img_dir(calib.path, calib.max_images)

        class DataLoader(pot.DataLoader):
            def __init__(self, calib: ImageCalibrationConfig):
                self.calib_cfg = calib
                super().__init__({})

            def __getitem__(self, index: int):
                image_path = files[index]
                if inp.shape is None:
                    exit_with(
                        ValueError(
                            "Input shape must be specified for calibration."
                        )
                    )
                img = read_image(
                    image_path,
                    inp.shape,  # type: ignore # TODO: dynamic shapes
                    inp.encoding.to,
                    self.calib_cfg.resize_method,
                    data_type=inp.data_type,
                )
                annotation = (index, np.zeros((1,)))
                img = np.expand_dims(img, axis=0).astype(np.float32)
                return annotation, img, {}

            def __len__(self):
                return len(files)

        model_config = addict.Dict(
            {
                "model_name": f"{xml_path.stem}-int8",
                "model": xml_path,
                "weights": xml_path.with_suffix(".bin"),
            }
        )

        engine_config = addict.Dict({"device": "CPU"})

        algorithms = [
            {
                "name": "DefaultQuantization",
                "stat_subset_size": 300,
                "params": {
                    "target_device": self.pot_target_device.name,
                    "preset": "performance",
                },
            }
        ]

        ir_model = pot.load_model(model_config=model_config)

        dataloader = DataLoader(calib)

        engine = pot.IEEngine(config=engine_config, data_loader=dataloader)

        pipeline = pot.create_pipeline(algorithms, engine)

        algorithm_name = pipeline.algo_seq[0].name
        preset = pipeline._algo_seq[0].config["preset"]

        logger.info(
            f'Executing POT pipeline on {model_config["model"]} '
            f"with {algorithm_name}, {preset} preset"
        )

        compressed_model = pipeline.run(ir_model)

        pot.compress_model_weights(compressed_model)

        compressed_model_path = pot.save_model(
            model=compressed_model,
            save_path=output_dir,
            model_name=ir_model.name,
        )[0]["model"]

        logger.info("Quantization finished successfully")
        return Path(compressed_model_path)
