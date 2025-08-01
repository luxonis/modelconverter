import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, NamedTuple, cast

from loguru import logger

from modelconverter.packages.base_exporter import Exporter
from modelconverter.utils import (
    ONNXModifier,
    exit_with,
    onnx_attach_normalization_to_inputs,
    read_image,
)
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.subprocess import subprocess_run
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    ResizeMethod,
    Target,
)


class RVC4Exporter(Exporter):
    target: Target = Target.RVC4

    def __init__(self, config: SingleStageConfig, output_dir: Path):
        # NOTE: Handled inside `snpe-onnx-to-dlc` command.
        config.disable_onnx_simplification = True

        super().__init__(config=config, output_dir=output_dir)

        rvc4_cfg = config.rvc4
        self.compress_to_fp16 = rvc4_cfg.compress_to_fp16
        self.snpe_onnx_to_dlc = rvc4_cfg.snpe_onnx_to_dlc_args
        self.snpe_dlc_quant = rvc4_cfg.snpe_dlc_quant_args
        self.snpe_dlc_graph_prepare = rvc4_cfg.snpe_dlc_graph_prepare_args
        self.use_per_channel_quantization = (
            rvc4_cfg.use_per_channel_quantization
        )
        self.use_per_row_quantization = rvc4_cfg.use_per_row_quantization
        self.optimization_level = rvc4_cfg.optimization_level
        self.keep_raw_images = rvc4_cfg.keep_raw_images
        if "--htp_socs" in self.snpe_dlc_graph_prepare:
            i = self.snpe_dlc_graph_prepare.index("--htp_socs")
            self.htp_socs = self.snpe_dlc_graph_prepare[i + 1].split(",")
        else:
            self.htp_socs = rvc4_cfg.htp_socs

        if self.config.input_file_type == InputFileType.ONNX:
            self.input_model = onnx_attach_normalization_to_inputs(
                self.input_model,
                self._attach_suffix(self.input_model, "modified.onnx"),
                self.inputs,
            )

            if not config.disable_onnx_optimization:
                onnx_modifier = ONNXModifier(
                    model_path=self.input_model,
                    output_path=self._attach_suffix(
                        self.input_model, "modified_optimized.onnx"
                    ),
                )

                try:
                    if (
                        onnx_modifier.modify_onnx()
                        and onnx_modifier.compare_outputs()
                    ):
                        logger.info("ONNX model has been optimized for RVC4.")
                        shutil.move(
                            onnx_modifier.output_path, self.input_model
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to optimize ONNX model: {e}. "
                        "Proceeding with unoptimized model."
                    )
                finally:
                    if onnx_modifier.output_path.exists():
                        onnx_modifier.output_path.unlink()
        else:
            logger.warning(
                "Input file type is not ONNX. Skipping pre-processing."
            )
        self.raw_img_dir = self.intermediate_outputs_dir / "raw_files"
        self.input_list_path = self.intermediate_outputs_dir / "img_list.txt"

    def export(self) -> Path:
        out_dlc_path = self.output_dir / f"{self.model_name}.dlc"
        self._inference_model_path = out_dlc_path

        dlc_path = self.onnx_to_dlc()
        if self._disable_calibration:
            quantized_dlc_path = dlc_path
        else:
            quantized_dlc_path = self.calibrate(dlc_path)

        logger.info("Performing offline graph preparation.")
        args = self.snpe_dlc_graph_prepare
        self._add_args(args, ["--input_dlc", quantized_dlc_path])
        self._add_args(args, ["--output_dlc", out_dlc_path])
        self._add_args(
            args,
            ["--set_output_tensors", ",".join(name for name in self.outputs)],
        )
        self._add_args(
            args, ["--optimization_level", str(self.optimization_level)]
        )
        self._add_args(args, ["--htp_socs", ",".join(self.htp_socs)])
        if self.compress_to_fp16:
            self._add_args(args, ["--use_float_io"])
        self._subprocess_run(
            ["snpe-dlc-graph-prepare", *args], meta_name="graph_prepare"
        )
        logger.info("Offline graph preparation finished.")
        self._inference_model_path = out_dlc_path
        subprocess_run(
            [
                "snpe-dlc-info",
                "-i",
                out_dlc_path,
                "-s",
                self.output_dir / "info.csv",
            ],
            silent=True,
        )
        return out_dlc_path

    def calibrate(self, dlc_path: Path) -> Path:
        args = self.snpe_dlc_quant
        if "--input_list" not in args:
            logger.info("Preparing calibration data.")
            calibration_list = self.prepare_calibration_data()
            if calibration_list is None:
                return dlc_path
            args.extend(["--input_list", str(calibration_list)])
        else:
            logger.info("Using provided `input_list`.")

        logger.info("Quantizing model.")
        quantized_dlc_path = self._attach_suffix(
            self.input_model, "quantized.dlc"
        )
        self._add_args(args, ["--input_dlc", dlc_path])
        self._add_args(args, ["--output_dlc", quantized_dlc_path])

        if self.use_per_channel_quantization:
            args.append("--use_per_channel_quantization")

        if self.use_per_row_quantization:
            args.append("--use_per_row_quantization")

        start_time = time.time()
        self._subprocess_run(
            ["snpe-dlc-quant", *args], meta_name="quantization_cmd"
        )

        logger.info(
            f"Quantization finished in {time.time() - start_time:.2f} seconds"
        )

        if not self.keep_raw_images and self.raw_img_dir.exists():
            shutil.rmtree(self.raw_img_dir)
            self.input_list_path.unlink()

        return quantized_dlc_path

    def prepare_calibration_data(self) -> Path | None:
        class Entry(NamedTuple):
            name: str
            path: Path
            encoding: Encoding
            resize_method: ResizeMethod
            shape: list[int]
            data_type: DataType

        entries: list[list[Entry]] = []

        for name, inp in self.inputs.items():
            calib = inp.calibration
            assert isinstance(calib, ImageCalibrationConfig)
            if inp.shape is None:
                exit_with(
                    ValueError(f"Input `{name}` has no shape specified.")
                )
            if not all(x is not None for x in inp.shape):
                exit_with(ValueError(f"Input `{name}` has dynamic shape."))
            shape = cast(list[int], inp.shape)
            if self.is_tflite:
                shape = [shape[0], shape[3], shape[1], shape[2]]
            entries.append(
                [
                    Entry(
                        name=name,
                        path=path,
                        encoding=inp.encoding.to,
                        resize_method=calib.resize_method,
                        shape=shape,
                        data_type=inp.data_type,
                    )
                    for path in self.read_img_dir(calib.path, calib.max_images)
                ]
            )

        if self.raw_img_dir.exists():
            logger.warning("Removing existing raw_images directory.")
            shutil.rmtree(self.raw_img_dir)
        self.raw_img_dir.mkdir(exist_ok=True)
        i = 0
        with open(self.input_list_path, "w") as f:
            log = True
            for entry in zip(*entries, strict=True):
                entry_str = ""
                for e in entry:
                    i += 1
                    if e.path.suffix == ".raw":
                        entry_str += f"{e.name}:={e.path} "
                    else:
                        img = read_image(
                            e.path,
                            shape=e.shape,
                            encoding=e.encoding,
                            resize_method=e.resize_method,
                            data_type=e.data_type,
                            transpose=False,
                        )
                        raw_path = self.raw_img_dir / f"{i}.raw"
                        img.tofile(raw_path)
                        entry_str += f"{e.name}:={raw_path} "
                entry_str = entry_str.strip()
                if log:
                    logger.debug(f"Image list entry: {entry_str}")
                    log = False
                f.write(entry_str + "\n")
        return self.input_list_path

    def onnx_to_dlc(self) -> Path:
        logger.info("Exporting for RVC4")
        args = self.snpe_onnx_to_dlc
        self._add_args(args, ["-i", self.input_model])
        if "--input_dim" not in args:
            for name, inp in self.inputs.items():
                if inp.shape is not None:
                    args.extend(
                        [
                            "--input_dim",
                            name,
                            ",".join(str(x) for x in inp.shape),
                        ]
                    )
        if "--input_dtype" not in args:
            for name, inp in self.inputs.items():
                if inp.data_type is not None:
                    args.extend(
                        ["--input_dtype", name, inp.data_type.as_snpe_dtype()]
                    )
        if "--out_name" not in args:
            for name in self.outputs:
                args.extend(["--out_name", name])

        if "--input_layout" not in args:
            for name, inp in self.inputs.items():
                layout = inp.layout
                if layout is None:
                    continue
                if layout in ["NCD", "NDC", "D"]:
                    layout = layout.replace("D", "F")
                if layout in [
                    "NCDHW",
                    "NDHWC",
                    "NCHW",
                    "NHWC",
                    "NFC",
                    "NCF",
                    "NTF",
                    "TNF",
                    "NF",
                    "NC",
                    "F",
                    "NONTRIVIAL",
                ]:
                    args.extend(["--input_layout", name, layout])
                else:
                    logger.warning(
                        f"Layout '{layout}' not supported by snpe for input '{name}'. "
                        "Proceeding wihtout specifying layout."
                    )

        if self.compress_to_fp16:
            self._add_args(args, ["--float_bitwidth", "16"])

        if self.is_tflite:
            command = "snpe-tflite-to-dlc"
        else:
            command = "snpe-onnx-to-dlc"
        self._subprocess_run([command, *args], meta_name="dlc_convert")
        logger.info("Exported for RVC4")
        return self.input_model.with_suffix(".dlc")

    def exporter_buildinfo(self) -> dict[str, Any]:
        snpe_version = subprocess.run(
            ["snpe-dlc-quant", "--version"], capture_output=True, check=False
        )
        return {
            "snpe_version": snpe_version.stdout.decode("utf-8").strip(),
            "target_devices": self.htp_socs,
        }
