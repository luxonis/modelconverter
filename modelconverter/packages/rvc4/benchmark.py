import io
import json
import re
import shlex
import shutil
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any, Final

import depthai as dai
import numpy as np
import polars as pl
from depthai import XLinkPlatform
from loguru import logger
from rich.progress import track

from modelconverter.packages.base_benchmark import Benchmark, Configuration
from modelconverter.utils import (
    DataType,
    DeviceMonitor,
    create_handler,
    create_progress_handler,
    environ,
    subprocess_run,
)
from modelconverter.utils.config import OutputConfig


class InputSpec(OutputConfig):
    shape: list[int]  # type: ignore


PROFILES: Final[list[str]] = [
    "low_balanced",
    "balanced",
    "high_performance",
    "sustained_high_performance",
    "burst",
    "low_power_saver",
    "power_saver",
    "high_power_saver",
    "extreme_power_saver",
    "system_settings",
]

RUNTIMES: dict[str, str] = {
    "dsp": "use_dsp",
    "cpu": "use_cpu",
}


class RVC4Benchmark(Benchmark):
    MAX_REAL_SNPE_INPUTS = 100

    @property
    def default_configuration(self) -> Configuration:
        """
        profile: The SNPE profile to use for inference.
        runtime: The SNPE runtime to use for inference.
        num_images: The number of images to use for inference.
        dai_benchmark: Whether to use the DepthAI for benchmarking.
        repetitions: The number of repetitions to perform (dai-benchmark only, ignored if benchmark_time is set).
        benchmark_time: Duration in seconds for time-based benchmarking (overrides repetitions).
        num_threads: The number of threads to use for inference (dai-benchmark only).
        num_messages: The number of messages to use for inference (dai-benchmark only).
        """
        return {
            "profile": "balanced",
            "runtime": "dsp",
            "num_images": 1000,
            "dai_benchmark": True,
            "repetitions": 10,
            "benchmark_time": 20,
            "num_threads": 2,
            "num_messages": 50,
            "device_ip": None,
            "device_id": None,
            "device_monitor": True,
        }

    @property
    def all_configurations(self) -> list[Configuration]:
        return [
            {"profile": profile, "num_threads": threads}
            for profile in PROFILES
            for threads in [1, 2]
        ]

    def _get_dlc_input_specs(
        self, model_path: str | Path | None = None
    ) -> list[InputSpec]:
        """Retrieve normalized input specs from a DLC or NNArchive."""
        model_path = self.model_path if model_path is None else model_path

        if str(model_path).endswith(".tar.xz"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                shutil.unpack_archive(model_path, tmp_dir)
                dlc_files = list(Path(tmp_dir).rglob("*.dlc"))
                if not dlc_files:
                    raise ValueError("No .dlc file found in the archive.")
                return self._get_dlc_input_specs(dlc_files[0])

        if not str(model_path).endswith(".dlc"):
            raise ValueError("Expected .dlc, or .tar.xz model format.")

        csv_path = Path("info.csv")
        subprocess_run(
            [
                "snpe-dlc-info",
                "-i",
                model_path,
                "-s",
                csv_path,
            ],
            silent=True,
        )
        content = csv_path.read_text()
        csv_path.unlink()

        start_marker = "Input Name,Dimensions,Type,Encoding Info"
        end_marker = "Output Name,Dimensions,Type,Encoding Info"
        start_index = content.find(start_marker)
        end_index = content.find(end_marker, start_index)

        # Extract and load the relevant CSV part into a polars DataFrame.
        relevant_csv_part = content[start_index:end_index].strip()
        df = pl.read_csv(io.StringIO(relevant_csv_part))
        df = df.with_columns(
            pl.col("Dimensions").str.split(",").cast(pl.List(pl.Int64))
        )

        rows = df.rows(named=True)

        return [
            InputSpec(
                name=str(row["Input Name"]),
                shape=list(row["Dimensions"]),
                data_type=DataType.from_dlc_dtype(str(row["Type"])),
            )
            for row in rows
        ]

    def _prepare_raw_inputs(
        self, input_specs: list[InputSpec], num_images: int
    ) -> None:
        if num_images < 1:
            raise ValueError("num_images must be at least 1.")

        model_dir = f"/data/modelconverter/{self.model_name}"
        inputs_dir = f"{model_dir}/inputs"
        real_input_count = min(num_images, self.MAX_REAL_SNPE_INPUTS)

        self.handler.shell(f"mkdir -p {model_dir}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_model_dir = Path(tmp_dir) / self.model_name
            local_inputs_dir = local_model_dir / "inputs"
            local_inputs_dir.mkdir(parents=True)
            local_input_list_path = local_model_dir / "input_list.txt"

            with local_input_list_path.open("w", encoding="utf-8") as f:
                for i in track(
                    range(num_images),
                    description="Preparing inputs",
                    total=min(num_images, self.MAX_REAL_SNPE_INPUTS),
                ):
                    input_paths: list[str] = []
                    for spec in input_specs:
                        input_path = f"{inputs_dir}/{spec.name}_{i}.raw"
                        input_paths.append(f"{spec.name}:={input_path}")

                        if i < real_input_count:
                            input_data = self._create_random_input(spec)
                            local_input_path = local_inputs_dir / (
                                f"{spec.name}_{i}.raw"
                            )
                            input_data.tofile(local_input_path)

                    f.write(" ".join(input_paths))
                    f.write(" \n")

            self.handler.push(local_model_dir, "/data/modelconverter")

        if num_images > real_input_count and input_specs:
            logger.info(
                f"Linking additional {num_images - real_input_count} inputs "
                f"to the first {real_input_count} inputs to avoid filling "
                "up the device storage."
            )
            self.handler.shell(
                self._create_raw_input_link_script(
                    input_specs,
                    inputs_dir,
                    real_input_count,
                    num_images,
                )
            )

    @staticmethod
    def _create_raw_input_link_script(
        input_specs: list[InputSpec],
        inputs_dir: str,
        real_input_count: int,
        num_images: int,
    ) -> str:
        spec_names = " ".join(shlex.quote(spec.name) for spec in input_specs)
        return "\n".join(
            [
                "set -eu",
                f"inputs_dir={shlex.quote(inputs_dir)}",
                f"real_input_count={real_input_count}",
                f"num_images={num_images}",
                f"for spec_name in {spec_names}; do",
                '    i="$real_input_count"',
                '    while [ "$i" -lt "$num_images" ]; do',
                "        source_index=$((i % real_input_count))",
                '        ln -f "$inputs_dir/${spec_name}_${source_index}.raw" "$inputs_dir/${spec_name}_${i}.raw"',
                "        i=$((i + 1))",
                "    done",
                "done",
            ]
        )

    @staticmethod
    def _create_random_input(spec: InputSpec) -> np.ndarray:
        if spec.data_type is DataType.INT32:
            # INT inputs are often token/index tensors;
            # zero keeps synthetic benchmark inputs in a safe range.
            return np.zeros(spec.shape, dtype=np.int32)
        if spec.data_type == DataType.INT8:
            return np.random.randint(-128, 128, size=spec.shape, dtype=np.int8)
        if spec.data_type == DataType.UFXP8:
            return np.random.randint(0, 256, size=spec.shape, dtype=np.uint8)

        return np.random.rand(*spec.shape).astype(
            spec.data_type.as_numpy_dtype()
        )

    def benchmark(self, configuration: Configuration) -> dict[str, Any]:
        dai_benchmark = configuration.get("dai_benchmark")
        device_monitor = configuration.get("device_monitor")

        device_ip, device_adb_id = get_device_info(
            configuration.get("device_ip"), configuration.get("device_id")
        )
        if device_monitor or not dai_benchmark:
            self.handler = create_handler(device_ip, device_adb_id)

        configuration["device_ip"] = device_ip

        self.monitor = None
        idle_measurements = {}
        if device_monitor:
            self.monitor = DeviceMonitor(self.handler)
            idle_measurements = self.monitor.get_idle_measurements()

        try:
            if dai_benchmark:
                for key in [
                    "dai_benchmark",
                    "num_images",
                    "device_id",
                    "device_monitor",
                ]:
                    configuration.pop(key)
                result = self._benchmark_dai(self.model_path, **configuration)
            else:
                for key in [
                    "dai_benchmark",
                    "repetitions",
                    "num_threads",
                    "num_messages",
                    "benchmark_time",
                    "device_ip",
                    "device_id",
                    "device_monitor",
                ]:
                    configuration.pop(key, None)
                logger.info("Running SNPE benchmark over ADB")
                result = self._benchmark_snpe(self.model_path, **configuration)

            if self.monitor:
                result |= self.monitor.get_stats()
                result |= idle_measurements
            return result
        finally:
            if self.monitor:
                self.monitor.stop()
            if not dai_benchmark:
                # so we don't delete the wrong directory
                # if `model_name` gets unset for any reason
                if not self.model_name:
                    raise AssertionError(
                        "`model_name` is not set, "
                        "cannot clean up model files on the device."
                    )

                self.handler.shell(
                    f"rm -rf /data/modelconverter/{self.model_name}"
                )

    def _benchmark_snpe(
        self,
        model_path: Path | str,
        num_images: int,
        profile: str,
        runtime: str,
    ) -> dict[str, Any]:
        runtime = RUNTIMES.get(runtime, "use_dsp")

        if isinstance(model_path, str) or str(model_path).endswith(".tar.xz"):
            if isinstance(model_path, str):
                model_archive = dai.getModelFromZoo(
                    dai.NNModelDescription(
                        model_path,
                        platform=dai.Platform.RVC4.name,
                    ),
                    apiKey=environ.HUBAI_API_KEY or "",
                )
            else:
                model_archive = model_path

            tmp_dir = Path(model_archive).parent / "tmp"
            shutil.unpack_archive(model_archive, tmp_dir)

            dlc_model_name = json.loads((tmp_dir / "config.json").read_text())[
                "model"
            ]["metadata"]["path"]
            dlc_path = next(tmp_dir.rglob(dlc_model_name), None)
            if not dlc_path:
                raise ValueError("Could not find model.dlc in the archive.")
            try:
                input_specs = self._get_dlc_input_specs(dlc_path)
            except Exception as e:
                logger.warning(
                    f"Failed to read input specs from the DLC "
                    f"with error: {e}. Reading from the archive."
                )
                input_specs = self._get_archive_input_specs(
                    dai.NNArchive(model_archive)
                )

        elif str(model_path).endswith(".dlc"):
            dlc_path = model_path
            input_specs = self._get_dlc_input_specs(dlc_path)

        else:
            raise ValueError(
                "Unsupported model format. Supported formats: .dlc, or HubAI model slug."
            )
        logger.info(
            f"Using SNPE profile '{profile}' and runtime '{runtime}' for benchmarking."
        )
        logger.info(f"Moving model '{dlc_path.name}' to the device.")

        self.handler.shell(f"mkdir -p /data/modelconverter/{self.model_name}")
        self.handler.push(
            str(dlc_path), f"/data/modelconverter/{self.model_name}/model.dlc"
        )
        self._prepare_raw_inputs(input_specs, num_images)

        if self.monitor:
            self.monitor.start()

        logger.info("Starting SNPE benchmark...")
        retcode, stdout, _ = self.handler.shell(
            # "source /data/modelconverter/source_me.sh && "
            "snpe-parallel-run "
            f"--container /data/modelconverter/{self.model_name}/model.dlc "
            f"--input_list /data/modelconverter/{self.model_name}/input_list.txt "
            f"--output_dir /data/modelconverter/{self.model_name}/outputs "
            f"--perf_profile {profile} "
            "--cpu_fallback true "
            f"--{runtime}",
            check=False,
        )
        if retcode == 137:
            raise RuntimeError(
                "Benchmark process was killed, likely due to out-of-memory. "
                "Consider decreasing the number of images (`--num-images`)."
            )
        pattern = re.compile(r"(\d+\.\d+) infs/sec")
        match = pattern.search(stdout)
        if match is None:
            raise RuntimeError(
                "Could not find throughput in stdout. Likely a server error.\n"
                f"stdout:\n{stdout}"
            )
        fps = float(match.group(1))
        return {"fps": fps, "latency": "N/A"}

    def _benchmark_dai(
        self,
        model_path: Path | str,
        profile: str,
        runtime: str,
        repetitions: int,
        num_threads: int,
        num_messages: int,
        benchmark_time: int,
        device_ip: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(model_path, str):
            resolved_model_path = Path(
                dai.getModelFromZoo(
                    dai.NNModelDescription(
                        model_path,
                        platform=dai.Platform.RVC4.name,
                    ),
                    apiKey=environ.HUBAI_API_KEY or "",
                )
            )
        elif str(model_path).endswith(".tar.xz"):
            resolved_model_path = Path(model_path)
        elif str(model_path).endswith(".dlc"):
            raise ValueError(
                "DLC model format is not currently supported for dai-benchmark. Please use SNPE for DLC models."
            )
        else:
            raise ValueError(
                "Unsupported model format. Supported formats: .tar.xz, or HubAI model slug."
            )

        model_archive = dai.NNArchive(resolved_model_path)
        try:
            logger.info("Trying to get input specs from the DLC file...")
            input_specs = self._get_dlc_input_specs(resolved_model_path)
        except Exception as e:
            logger.warning(
                f"Failed to read input specs from the DLC with error: {e}"
            )
            logger.info("Reading specs from the archive.")
            input_specs = self._get_archive_input_specs(model_archive)

        input_data_packet = dai.NNData()
        for spec in input_specs:
            input_data = self._create_random_input(spec)
            input_data_packet.addTensor(
                spec.name, input_data, dataType=spec.data_type.as_dai_dtype()
            )

        if device_ip:
            device = dai.Device(dai.DeviceInfo(device_ip))
        else:
            for info in dai.Device.getAllAvailableDevices():
                if info.platform == XLinkPlatform.X_LINK_RVC4:
                    device = dai.Device(info)
                    break
            else:
                raise RuntimeError("No RVC4 device found.")

        if device.getPlatform() != dai.Platform.RVC4:
            raise ValueError(
                f"Found {device.getPlatformAsString()}, expected RVC4 platform."
            )

        logger.info(
            f"Using {device.getPlatformAsString()} device on IP {device.getDeviceInfo().name}."
        )

        if self.monitor:
            self.monitor.start()

        with dai.Pipeline(device) as pipeline:
            benchmark_out = pipeline.create(dai.node.BenchmarkOut)
            benchmark_out.setRunOnHost(False)
            benchmark_out.setFps(-1)

            neural_network = pipeline.create(dai.node.NeuralNetwork)
            neural_network.setNNArchive(model_archive)

            neural_network.setBackendProperties(
                {
                    "runtime": runtime,
                    "performance_profile": profile,
                }
            )

            neural_network.setNumInferenceThreads(num_threads)

            benchmark_in = pipeline.create(dai.node.BenchmarkIn)
            benchmark_in.setRunOnHost(False)
            benchmark_in.sendReportEveryNMessages(num_messages)
            benchmark_in.logReportsAsWarnings(False)

            benchmark_out.out.link(neural_network.input)
            neural_network.out.link(benchmark_in.input)

            output_queue = benchmark_in.report.createOutputQueue()
            input_queue = benchmark_out.input.createInputQueue()

            pipeline.start()
            input_queue.send(input_data_packet)

            progress, on_tick, should_continue = create_progress_handler(
                benchmark_time, repetitions
            )

            fps_list = []
            avg_latency_list = []

            with progress:
                while pipeline.isRunning() and should_continue():
                    benchmark_report = output_queue.get()
                    if not isinstance(benchmark_report, dai.BenchmarkReport):
                        raise TypeError(
                            f"Expected BenchmarkReport, got {type(benchmark_report)}"
                        )

                    fps_list.append(benchmark_report.fps)
                    avg_latency_list.append(
                        benchmark_report.averageLatency * 1000
                    )

                    on_tick()

            # Currently, the latency measurement is only supported on RVC4 when using ImgFrame as the input to the BenchmarkOut which we don't do here.
            return {"fps": float(np.mean(fps_list)), "latency": "N/A"}

    def _extra_header(
        self,
        results: list[tuple[Configuration, dict[str, Any]]],
    ) -> list[str]:
        heads = []
        if self.monitor:
            heads.append("power_sys (W)")
            heads.append("power_core (W)")
            heads.append("dsp (%)")
            heads.append("memory (MiB)")
            heads.append("cpu (%)")
        return heads

    def _extra_row_cells(
        self,
        configuration: Configuration,
        result: dict[str, Any],
    ) -> Iterable[str]:
        power_sys = result.get("power_system")
        power_core = result.get("power_processor")
        dsp = result.get("dsp_utilization")
        memory = result.get("ram_used")
        cpu = result.get("cpu_utilization")

        if self.monitor:
            yield f"{power_sys:.2f}" if power_sys else "[orange3]N/A"
            yield f"{power_core:.2f}" if power_core else "[orange3]N/A"
            yield f"{dsp:.2f}" if dsp else "[orange3]N/A"
            yield f"{memory:.2f}" if memory else "[orange3]N/A"
            yield f"{cpu:.2f}" if cpu else "[orange3]N/A"

    def _get_archive_input_specs(
        self, archive: dai.NNArchive
    ) -> list[InputSpec]:
        def guess_dtype(
            name: str,
            input_precision: DataType,
            archive_precision: DataType | None,
            hubai_precision: DataType | None = None,
        ) -> DataType:
            logger.info(f"Resolving correct type for input '{name}'")
            logger.info(f"model.inputs.{name}.dtype = {input_precision}")
            logger.info(f"model.metadata.precision = {archive_precision}")
            if hubai_precision is not None:
                logger.info(f"HubAI is reporting {hubai_precision}")

            result = None

            # e.g. for inputs representing indices
            if input_precision not in {
                DataType.INT8,
                DataType.FLOAT16,
                DataType.FLOAT32,
            }:
                result = input_precision
                logger.info("Using the model.inputs value")
            elif hubai_precision is not None:
                logger.info("Using the HubAI value")
                result = hubai_precision
            else:
                match archive_precision, input_precision:
                    case (
                        DataType.INT8 | None,
                        DataType.INT8 | DataType.FLOAT32,
                    ):
                        result = DataType.INT8
                    case (
                        DataType.FLOAT16 | None,
                        DataType.FLOAT16 | DataType.FLOAT32,
                    ):
                        result = DataType.FLOAT16
                    case DataType.FLOAT32 | None, DataType.FLOAT32:
                        result = DataType.FLOAT32
                    case _:
                        result = input_precision
            if result is None:
                raise ValueError("Unable to resolve the type combination")
            logger.info(
                f"Resolved the type of '{name}' to be '{result.name.upper()}'"
            )
            return result

        cfg = archive.getConfig()
        return [
            InputSpec(
                name=inp.name,
                shape=inp.shape,
                data_type=guess_dtype(
                    inp.name,
                    DataType(inp.dtype.name.lower()),
                    archive_precision=DataType(
                        cfg.model.metadata.precision.name.lower()
                    )
                    if cfg.model.metadata.precision
                    else None,
                    hubai_precision=self._get_hubai_type(),
                ),
            )
            for inp in cfg.model.inputs
        ]

    def _get_hubai_type(self) -> DataType | None:
        from modelconverter.cli import Request, slug_to_id

        if not isinstance(
            self.model_path, str
        ) or not self.HUB_MODEL_PATTERN.match(self.model_path):
            return None

        model_id = slug_to_id(self.model_name, "models")
        model_variant = self.model_path.split(":")[1]

        model_variants = []
        for is_public in [True, False, None]:
            with suppress(Exception):
                model_variants += Request.get(
                    "modelVersions/",
                    params={"model_id": model_id, "is_public": is_public},
                )

        model_version_id = None
        for version in model_variants:
            if version["variant_slug"] == model_variant:
                model_version_id = version["id"]
                break

        if not model_version_id:
            return DataType.INT8

        model_instances = []
        for is_public in [True, False]:
            with suppress(Exception):
                model_instances += Request.get(
                    "modelInstances/",
                    params={
                        "model_id": model_id,
                        "model_version_id": model_version_id,
                        "is_public": is_public,
                    },
                )

        model_precision_type = None
        for instance in model_instances:
            if instance["platforms"] == ["RVC4"] and (
                self.model_instance is None
                or instance["hash_short"] == self.model_instance
            ):
                model_precision_type = instance.get("model_precision_type")
                break

        if model_precision_type is not None:
            return DataType.from_hubai_dtype(model_precision_type)
        return None


def device_id_to_adb_id(device_id: str) -> str:
    if device_id.isdigit():
        return format(int(device_id), "x")
    return device_id.encode("ascii").hex()


def adb_id_to_device_id(adb_id: str) -> str:
    try:
        int_id = int(adb_id, 16)
        return str(int_id)
    except ValueError:
        bytes_id = bytes.fromhex(adb_id)
        return bytes_id.decode("ascii")


def get_device_info(
    device_ip: str | None, device_id: str | None
) -> tuple[str | None, str | None]:
    if not device_ip and not device_id:
        return None, None

    if device_id:
        if device_id.isdecimal():
            adb_id = device_id_to_adb_id(device_id)
        else:
            adb_id = device_id
            device_id = adb_id_to_device_id(adb_id)
        for info in dai.Device.getAllAvailableDevices():
            if device_id == info.getDeviceId():
                if device_ip and device_ip != info.name:
                    logger.warning(
                        f"Both device_id and device_ip provided, but they refer to different devices. Using device with device_id: {device_id} and device_ip: {info.name}."
                    )
                return info.name, adb_id
    if device_ip:
        with dai.Device(device_ip) as device:
            inferred_device_id = device.getDeviceId()
            return device_ip, device_id_to_adb_id(inferred_device_id)
    return None, None
