"""Export of models to the RVC4 platform.

Drives the Qualcomm SNPE toolchain shipped in the RVC4 Docker image:
``snpe-onnx-to-dlc`` (or ``snpe-tflite-to-dlc``) produces a DLC model,
``snpe-dlc-quant`` quantizes it against the calibration data, and
``snpe-dlc-graph-prepare`` builds the offline graph for the targeted HTP
SoCs. Which SNPE arguments are used follows from the configured
quantization mode unless that mode is ``CUSTOM``.
"""

import json
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
    Encodings,
    ImageCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.subprocess import subprocess_run
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    QuantizationMode,
    ResizeMethod,
    Target,
)


class RVC4Exporter(Exporter):
    """Exporter producing a DLC model for the RVC4 platform."""

    target: Target = Target.RVC4

    def __init__(self, config: SingleStageConfig, output_dir: Path):
        """Initialize the exporter and pre-process the input model.

        The SNPE arguments and the quantization options are taken from
        the ``rvc4`` section of the config. Unless the quantization mode
        is ``CUSTOM``, the user-provided SNPE arguments are dropped in
        favor of the ones the mode implies. An ONNX input model gets the
        configured normalization attached to its inputs and, unless all
        ONNX optimizations are disabled, is rewritten by `ONNXModifier`
        -- the rewritten model is only kept when its outputs still match
        those of the original.

        Args:
            config: Configuration of the stage to export.
            output_dir: Directory the exported model is written to.

        """
        super().__init__(config=config, output_dir=output_dir)

        rvc4_cfg = config.rvc4
        self.encodings = rvc4_cfg.encodings
        self.snpe_onnx_to_dlc = rvc4_cfg.snpe_onnx_to_dlc_args
        self.snpe_dlc_quant = rvc4_cfg.snpe_dlc_quant_args
        self.snpe_dlc_graph_prepare = rvc4_cfg.snpe_dlc_graph_prepare_args
        self.use_per_channel_quantization = (
            rvc4_cfg.use_per_channel_quantization
        )
        self.use_per_row_quantization = rvc4_cfg.use_per_row_quantization
        self.optimization_level = rvc4_cfg.optimization_level
        self.quantization_mode = rvc4_cfg.quantization_mode
        if self.quantization_mode != QuantizationMode.CUSTOM:
            self.snpe_onnx_to_dlc = []
            self.snpe_dlc_quant = []
            self.snpe_dlc_graph_prepare = []
            logger.warning(
                f"Using pre-defined arguments for quantization mode {self.quantization_mode.value}, which will override user-provided SNPE arguments. If you need full control of SNPE arguments, set `rvc4.quantization_mode: CUSTOM` in the config or CLI."
            )
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

            if not self.onnx_optimizations.all_disabled():
                onnx_modifier = ONNXModifier(
                    model_path=self.input_model,
                    output_path=self._attach_suffix(
                        self.input_model, "modified_optimized.onnx"
                    ),
                )

                try:
                    if (
                        onnx_modifier.modify_onnx(
                            **self.onnx_optimizations.model_dump()
                        )
                        and onnx_modifier.compare_outputs()
                    ):
                        logger.info("ONNX model has been optimized for RVC4.")
                        shutil.move(
                            onnx_modifier.output_path, self.input_model
                        )
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        f"Failed to optimize ONNX model: {e}. "
                        "Proceeding with unoptimized model."
                    )
                finally:
                    if onnx_modifier.output_path.exists():  # pragma: no cover
                        onnx_modifier.output_path.unlink()
        else:
            logger.warning(
                "Input file type is not ONNX. Skipping pre-processing."
            )
        self.raw_img_dir = self.intermediate_outputs_dir / "raw_files"
        self.input_list_path = self.intermediate_outputs_dir / "img_list.txt"

    def export(self) -> Path:
        """Convert, quantize and graph-prepare the model.

        The input model is converted to DLC, quantized unless
        calibration is disabled, and prepared for the configured HTP
        SoCs at the configured optimization level. An ``info.csv``
        dump of the resulting model is written next to it.

        Returns:
            Path to the exported DLC model.

        """
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
        if self.quantization_mode == QuantizationMode.FP16_STD:
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
        """Quantize a DLC model with ``snpe-dlc-quant``.

        The calibration data is prepared from the config unless the SNPE
        arguments already name an ``--input_list``. Which quantizers and
        bit widths are used follows from the quantization mode. The raw
        images produced for the calibration are removed afterwards
        unless ``keep_raw_images`` is set.

        Args:
            dlc_path: Path to the DLC model to quantize.

        Returns:
            Path to the quantized DLC model.

        """
        args = self.snpe_dlc_quant
        if "--input_list" not in args:
            logger.info("Preparing calibration data.")
            calibration_list = self.prepare_calibration_data()
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

        if self.quantization_mode == QuantizationMode.INT8_ACC:
            self._add_args(args, ["--param_quantizer", "enhanced"])
            self._add_args(args, ["--act_quantizer", "enhanced"])
        elif self.quantization_mode == QuantizationMode.INT8_16_MIX_ACC:
            self._add_args(args, ["--param_quantizer", "enhanced"])
            self._add_args(args, ["--act_quantizer", "enhanced"])
            self._add_args(args, ["--act_bitwidth", "16"])
        elif self.quantization_mode == QuantizationMode.INT8_16_MIX:
            self._add_args(args, ["--act_bitwidth", "16"])

        if self.encodings is not None:
            args.append("--override_params")

        start_time = time.time()
        self._subprocess_run(
            ["snpe-dlc-quant", *args], meta_name="quantization_cmd"
        )

        logger.info(
            f"Quantization finished in {time.time() - start_time:.2f} seconds"
        )

        if (
            not self.keep_raw_images and self.raw_img_dir.exists()
        ):  # pragma: no cover
            shutil.rmtree(self.raw_img_dir)
            self.input_list_path.unlink()

        return quantized_dlc_path

    def prepare_calibration_data(self) -> Path:
        """Write the calibration data as raw files and list them.

        Every calibration file of every input is read with the input's
        encoding, resize method and data type and written out as a raw
        file, a ``.raw`` file being taken as it is. The SNPE input list
        holds one line per calibration sample, pairing each input name
        with its raw file. Terminates the process if an input has no
        shape or a dynamic one.

        Returns:
            Path to the written input list.

        """

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

        if self.raw_img_dir.exists():  # pragma: no cover
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

    def generate_io_encodings(self, encodings: Encodings) -> Path:
        """Write the quantization overrides to a JSON file.

        The encodings of the model's own inputs and outputs are replaced
        by the default 8-bit integer encoding, as DAI does not support
        custom TF8 encodings on exposed tensors. The encodings of the
        internal tensors are written out unchanged.

        Args:
            encodings: Encodings as resolved from the configuration,
                before the exposed tensors are normalized.

        Returns:
            Path to the written ``encodings.json``.

        """
        encodings_dict = encodings.model_dump(mode="json", exclude_none=True)
        # DAI does not support custom TF8 encodings on exposed tensors.
        # Keep AIMET's internal tensor encodings, but normalize exposed
        # inputs and outputs to default int8 IO.
        for name in list(self.inputs.keys()) + list(self.outputs.keys()):
            encodings_dict["activation_encodings"][name] = [
                {"bitwidth": 8, "dtype": "int"}
            ]
        encodings_path = self.intermediate_outputs_dir / "encodings.json"
        with open(encodings_path, "w") as encodings_file:
            json.dump(encodings_dict, encodings_file, indent=4)
        return encodings_path

    def onnx_to_dlc(self) -> Path:
        """Convert the input model to the DLC format.

        Runs ``snpe-onnx-to-dlc``, or ``snpe-tflite-to-dlc`` for a
        TFLite input. Input shapes, input data types, input layouts and
        output names are added to the SNPE arguments, each group only
        when the configured arguments do not already carry the
        corresponding flag. A layout SNPE does not know is skipped with
        a warning.

        Returns:
            Path to the converted DLC model.

        """
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
                # A converting input always has a shape (read from the model),
                # and the config derives a layout from any shape, so `layout`
                # is only ever None for a shapeless input that can't convert.
                if layout is None:  # pragma: no cover
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
                        "Proceeding without specifying layout."
                    )

        if self.quantization_mode == QuantizationMode.FP16_STD:
            self._add_args(args, ["--float_bitwidth", "16"])
        elif self.encodings is not None:
            io_encodings_file = self.generate_io_encodings(self.encodings)
            self._add_args(
                args,
                [
                    "--quantization_overrides",
                    f"{io_encodings_file}",
                ],
            )

        if self.is_tflite:
            command = "snpe-tflite-to-dlc"
        else:
            command = "snpe-onnx-to-dlc"
        self._subprocess_run([command, *args], meta_name="dlc_convert")
        logger.info("Exported for RVC4")
        return self.input_model.with_suffix(".dlc")

    def exporter_buildinfo(self) -> dict[str, Any]:
        """Collect the RVC4-specific build information.

        Returns:
            Dictionary with the version of the SNPE tooling the model
            was exported with and the HTP SoCs it was prepared for.

        """
        snpe_version = subprocess.run(
            ["snpe-dlc-quant", "--version"], capture_output=True, check=False
        )
        return {
            "snpe_version": snpe_version.stdout.decode("utf-8").strip(),
            "target_devices": self.htp_socs,
        }
