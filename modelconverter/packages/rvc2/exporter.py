import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from concurrent.futures import Future
from multiprocessing import cpu_count
from os import environ as env
from pathlib import Path
from threading import Lock
from typing import Any, Final, NamedTuple

import tflite2onnx
from loguru import logger

from modelconverter.packages.base_exporter import Exporter
from modelconverter.utils import (
    ONNXModifier,
    SubprocessHandle,
    get_container_memory_available,
    onnx_attach_normalization_to_inputs,
)
from modelconverter.utils.config import SingleStageConfig
from modelconverter.utils.subprocess import SubprocessResult
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    Target,
)

OV_2021: Final[bool] = env.get("VERSION") == "2021.4.0"

DEFAULT_SUPER_SHAVES: Final[int] = 8


class CompileResult(NamedTuple):
    blob_path: Path
    result: SubprocessResult


class RVC2Exporter(Exporter):
    target: Target = Target.RVC2

    def __init__(self, config: SingleStageConfig, output_dir: Path):
        super().__init__(config=config, output_dir=output_dir)
        self.n_workers = config.rvc2.n_workers
        self.compress_to_fp16 = config.rvc2.compress_to_fp16
        self.number_of_shaves = config.rvc2.number_of_shaves
        # Setting the number_of_cmx_slices equal to the number_of_shaves is intentional from the DAI's perspective to maximize RVC2 pipeline performance.
        self.number_of_cmx_slices = config.rvc2.number_of_shaves
        self.superblob = config.rvc2.superblob
        self.mo_args = config.rvc2.mo_args
        self.compile_tool_args = config.rvc2.compile_tool_args
        self.device = "MYRIAD"
        self.reverse_input_channels = False

        self._device_specific_buildinfo = {
            "is_superblob": self.superblob,
            "number_of_shaves": self.number_of_shaves
            if not self.superblob
            else list(range(1, 17)),
            "number_of_cmx_slices": self.number_of_cmx_slices,
        }

    def _export_openvino_ir(self) -> Path:
        args = self.mo_args
        self._add_args(args, ["--output_dir", self.intermediate_outputs_dir])
        self._add_args(args, ["--output", ",".join(self.outputs)])
        if self.compress_to_fp16:
            if OV_2021:
                self._add_args(args, ["--data_type", "FP16"])
            else:
                self._add_args(args, ["--compress_to_fp16"])

        if "--input" not in args:
            inp_str = ""
            for name, inp in self.inputs.items():
                if inp_str:
                    inp_str += ","
                inp_str += name
                if inp.shape is not None:
                    inp_str += f"{_lst_join(inp.shape, sep=' ')}"
                if inp.data_type is not None:
                    if OV_2021 and self.compress_to_fp16:
                        data_type = DataType("float16")
                    else:
                        data_type = inp.data_type
                    inp_str += f"{{{data_type.as_openvino_dtype()}}}"
                if inp.frozen_value is not None:
                    if len(inp.frozen_value) == 1:
                        value = inp.frozen_value[0]
                    else:
                        value = f"{_lst_join(inp.frozen_value)}"
                    inp_str += f"->{value}"
            args.extend(["--input", inp_str])

        if not self._check_reverse_channels():
            logger.warning(
                "The model optimizer does not support reversing "
                "input channels for only some inputs. "
                "Attempting to modify the ONNX model."
            )
            self.input_model = onnx_attach_normalization_to_inputs(
                self.input_model,
                self._attach_suffix(self.input_model, "modified.onnx"),
                self.inputs,
                reverse_only=True,
            )
            for inp in self.inputs.values():
                if inp.mean_values is not None and inp.encoding_mismatch:
                    inp.mean_values = inp.mean_values[::-1]
                if inp.scale_values is not None and inp.encoding_mismatch:
                    inp.scale_values = inp.scale_values[::-1]
                inp.encoding.from_ = Encoding.BGR
                inp.encoding.to = Encoding.BGR

        if not self.config.disable_onnx_optimization:
            onnx_modifier = ONNXModifier(
                model_path=self.input_model,
                output_path=self._attach_suffix(
                    self.input_model, "modified_optimized.onnx"
                ),
                skip_optimization=True,
            )

            try:
                if (
                    onnx_modifier.modify_onnx()
                    and onnx_modifier.compare_outputs()
                ):
                    logger.info("ONNX model has been optimized for RVC2.")
                    shutil.move(onnx_modifier.output_path, self.input_model)
            except Exception as e:
                logger.warning(
                    f"Failed to optimize ONNX model: {e}. "
                    "Proceeding with unoptimized model."
                )
            finally:
                if onnx_modifier.output_path.exists():
                    onnx_modifier.output_path.unlink()

        mean_values_str = ""
        scale_values_str = ""
        for name, inp in self.inputs.items():
            # Append mean values in a similar style
            if inp.is_color_input and inp.mean_values is not None:
                if mean_values_str:
                    mean_values_str += ","
                mean_values_str += (
                    f"{name}[{','.join(str(v) for v in inp.mean_values)}]"
                )

            # Append scale values in a similar style
            if inp.is_color_input and inp.scale_values is not None:
                if scale_values_str:
                    scale_values_str += ","
                scale_values_str += (
                    f"{name}[{','.join(str(v) for v in inp.scale_values)}]"
                )
        # Extend args with mean and scale values if they were collected
        if mean_values_str:
            args.extend(["--mean_values", mean_values_str])
        if scale_values_str:
            args.extend(["--scale_values", scale_values_str])

        # Append reverse_input_channels flag only once if needed
        reverse_input_flag = any(
            inp.encoding_mismatch for inp in self.inputs.values()
        )
        if reverse_input_flag:
            self.reverse_input_channels = True
            args.append("--reverse_input_channels")

        self._add_args(args, ["--input_model", self.input_model])

        self._subprocess_run(["mo", *args], meta_name="model_optimizer")

        logger.info(f"OpenVINO IR exported to {self.output_dir}")
        return self.input_model.with_suffix(".xml")

    def _check_reverse_channels(self) -> bool:
        reverses = [inp.encoding_mismatch for inp in self.inputs.values()]
        return all(reverses) or not any(reverses)

    @staticmethod
    def _write_config(shaves: int, slices: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".conf", delete=False) as conf:
            if not OV_2021:
                conf.write(b"MYRIAD_ENABLE_MX_BOOT NO\n")
            conf.write(f"MYRIAD_NUMBER_OF_SHAVES {shaves}\n".encode())
            conf.write(f"MYRIAD_NUMBER_OF_CMX_SLICES {slices}\n".encode())
            conf.write(b"MYRIAD_THROUGHPUT_STREAMS 1\n")
        return conf.name

    def _transform_tflite_to_onnx(self) -> None:
        logger.info("Converting TFLite model to ONNX.")
        logger.warning("The TFLite to ONNX conversion is experimental.")

        onnx_path = self.input_model.with_suffix(".onnx")
        tflite2onnx.convert(str(self.input_model), str(onnx_path))

        self.input_model = onnx_path
        self.input_file_type = InputFileType.ONNX

        for name, inp in self.inputs.items():
            if (
                inp.encoding.from_ == Encoding.NONE
                or not inp.layout
                or not inp.shape
            ):
                continue

            lt = inp.layout
            sh = inp.shape

            if lt[-1] == "C":
                if len(lt) == 4 and lt[0] == "N":
                    if not OV_2021:
                        self._add_args(
                            self.mo_args, ["--layout", f"{name}(nchw->nhwc)"]
                        )
                    inp.shape = [sh[0], sh[3], sh[1], sh[2]]
                    inp.layout = f"{lt[0]}{lt[3]}{lt[1]}{lt[2]}"

                elif len(inp.layout) == 3:
                    if not OV_2021:
                        self._add_args(
                            self.mo_args, ["--layout", f"{name}(chw->hwc)"]
                        )
                    inp.shape = [sh[2], sh[0], sh[1]]
                    inp.layout = f"{lt[2]}{lt[0]}{lt[1]}"

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
        if "-iop" not in args:
            self._add_args(args, ["-ip", "U8"])

        args += ["-m", xml_path]

        if self.superblob:
            return self.compile_superblob(args)

        return self.compile_blob(args).blob_path

    def compile_blob(self, args: list) -> CompileResult:
        output_path = (
            self.output_dir / f"{self.model_name}-{self.target.name.lower()}"
        )

        if "-o" not in args:
            blob_output_path = output_path.with_suffix(".blob")
            args += ["-o", blob_output_path]
        else:
            blob_output_path = Path(args[args.index("-o") + 1])

        if "-c" not in args:
            args += [
                "-c",
                self._write_config(
                    self.number_of_shaves, self.number_of_cmx_slices
                ),
            ]

        result = self._subprocess_run(
            ["compile_tool", *args], meta_name="compile_tool"
        )
        logger.info(f"Blob compiled to {blob_output_path}")
        return CompileResult(blob_output_path, result)

    def compile_superblob(self, args: list[str]) -> Path:
        from pebble import ThreadPool

        blobs_directory = self.intermediate_outputs_dir / "blobs"
        blobs_directory.mkdir(parents=True, exist_ok=True)

        orig_args = args.copy()

        # Base blob compile to get peak RAM usage
        default_blob_path, result = self.compile_blob(
            [
                *orig_args,
                "-o",
                blobs_directory
                / f"{self.model_name}_{DEFAULT_SUPER_SHAVES}shave.blob",
                "-c",
                self._write_config(DEFAULT_SUPER_SHAVES, DEFAULT_SUPER_SHAVES),
            ]
        )

        logger.info("Compiling superblob.")

        base_peak_mem = result.peak_memory * 1.5
        peaks = [base_peak_mem]

        avail_ram = get_container_memory_available() * 0.8
        n_workers = int(
            max(1, min(cpu_count(), 15, avail_ram // base_peak_mem - 1))
        )
        if self.n_workers is not None:
            if self.n_workers > n_workers:
                logger.warning(
                    f"Requested {self.n_workers} workers, which is "
                    f"more than the recommended {n_workers} based on "
                    "available RAM. This may lead to out-of-memory errors."
                )
            n_workers = self.n_workers

        logger.info(f"Available RAM: {avail_ram // 1e9} GB")
        logger.info(f"Base compile peak: {base_peak_mem // 1e6} MB")
        logger.info(f"Using {n_workers} workers for superblob compilation")

        # Order to compile patches to minimize RAM usage spikes
        shave_list = [1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10, 9]

        lock = Lock()

        def _superblob_compile_step(shaves: int) -> None:
            import bsdiff4

            args = orig_args.copy()
            args += ["-c", RVC2Exporter._write_config(shaves, shaves)]
            blob_path = (
                blobs_directory / f"{self.model_name}_{shaves}shave.blob"
            )
            default_blob_path = (
                blobs_directory
                / f"{self.model_name}_{DEFAULT_SUPER_SHAVES}shave.blob"
            )
            args += ["-o", blob_path]

            logger.info(f"Compiling {shaves}-shave patch...")
            with SubprocessHandle(
                ["compile_tool", *args], silent=True
            ) as handle:
                while handle.poll() is None:
                    if handle.is_suspended():
                        with lock:
                            avail_ram = get_container_memory_available() * 0.8
                            if avail_ram > max(peaks):
                                handle.resume()
                                # Give it a moment to ramp up so other
                                # threads don't all resume at once
                                # causing possible OOM
                                logger.info(
                                    f"Resuming {shaves}-shave compile, "
                                    f"available RAM is now "
                                    f"{avail_ram // 1e6} MB > "
                                    f"{max(peaks) // 1e6} MB"
                                )
                                time.sleep(5)
                    else:
                        with lock:
                            avail_ram = get_container_memory_available() * 0.8
                        if avail_ram < max(peaks):
                            logger.warning(
                                f"Suspending {shaves}-shave compile due to "
                                f"low available RAM ({avail_ram // 1e6} MB < "
                                f"{max(peaks) // 1e6} MB)"
                            )
                            handle.suspend()
                        time.sleep(0.5)
                res = handle.result()

            patch_file = blob_path.with_suffix(".patch")
            bsdiff4.file_diff(default_blob_path, blob_path, patch_file)
            peaks.append(res.peak_memory * 1.5)
            logger.info(
                f"Compiled {shaves}-shave patch in {res.total_time:.2f} s, "
                f"peak RAM {res.peak_memory // 1e6} MB"
            )

        t = time.time()
        futures: set[Future] = set()
        with ThreadPool(n_workers) as executor:
            for shave in shave_list:
                futures.add(
                    executor.submit(_superblob_compile_step, shaves=shave)
                )
                time.sleep(0.2)
            while any(not f.done() for f in futures):
                time.sleep(0.5)

        logger.info(f"Superblob compiled in {time.time() - t:.2f} seconds.")

        superblob_path = self.output_dir / f"{self.model_name}.superblob"

        idx2patch = {
            int(patch_name.stem.split("_")[-1][:-5]): patch_name
            for patch_name in blobs_directory.glob("*.patch")
        }

        patch2idx = {patch: idx for idx, patch in idx2patch.items()}

        with open(superblob_path, "wb") as superblob_file:
            header = default_blob_path.stat().st_size.to_bytes(8, "big")
            for patch_idx in range(1, 17):
                patchsize = (
                    idx2patch[patch_idx].stat().st_size
                    if patch_idx in idx2patch
                    else 0
                )
                header += patchsize.to_bytes(8, "big")
            superblob_file.write(header)

            with open(default_blob_path, "rb") as default_blob_file:
                superblob_file.write(default_blob_file.read())

            patches = sorted(patch2idx.keys(), key=lambda x: patch2idx[x])
            if not patches:
                raise RuntimeError("No patches found.")

            for patch_path in patches:
                with open(patch_path, "rb") as patch_file:
                    superblob_file.write(patch_file.read())

        logger.info(f"Superblob compiled to {superblob_path}")
        return superblob_path

    def exporter_buildinfo(self) -> dict[str, Any]:
        mo_version = (
            subprocess.run(
                ["mo", "--version"], capture_output=True, check=False
            )
            .stdout.decode()
            .split(":", 1)[1]
            .strip()
        )
        compile_tool_version, compile_tool_build = (
            subprocess.run(["compile_tool"], capture_output=True, check=False)
            .stdout.decode()
            .splitlines()[:2]
        )
        compile_tool_version = compile_tool_version.strip().split(" ")[-1]
        compile_tool_build = compile_tool_build.strip().split(" ")[-1]

        return {
            "model_optimizer_version": mo_version,
            "compile_tool_version": compile_tool_version,
            "compile_tool_build": compile_tool_build,
            "target_devices": [self.device],
            **self._device_specific_buildinfo,
        }


def _lst_join(args: Iterable[Any], sep: str = ",") -> str:
    return f"[{sep.join(map(str, args))}]"
