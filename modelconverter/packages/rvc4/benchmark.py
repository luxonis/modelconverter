import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, cast

import depthai as dai
import numpy as np
import pandas as pd
from loguru import logger
from rich.progress import Progress

from modelconverter.utils import environ, subprocess_run

from ..base_benchmark import Benchmark, BenchmarkResult, Configuration

PROFILES: Final[List[str]] = [
    "low_balanced",
    "balanced",
    "default",
    "high_performance",
    "sustained_high_performance",
    "burst",
    "low_power_saver",
    "power_saver",
    "high_power_saver",
    "extreme_power_saver",
    "system_settings",
]

RUNTIMES: Dict[str, str] = {
    "dsp": "use_dsp",
    "cpu": "use_cpu",
}


class AdbHandler:
    def __init__(self, device_id: Optional[str] = None) -> None:
        self.device_args = ["-s", device_id] if device_id else []

    def _adb_run(self, args, **kwargs) -> Tuple[int, str, str]:
        result = subprocess.run(
            ["adb", *self.device_args, *args],
            **kwargs,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout = result.stdout
        stderr = result.stderr
        assert result.returncode is not None
        if result.returncode != 0:
            raise RuntimeError(
                f"adb command {args[0]} failed with code {result.returncode}:\n"
                f"args: {args}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}\n"
            )
        return (
            result.returncode,
            stdout.decode(errors="ignore"),
            stderr.decode(errors="ignore"),
        )

    def shell(self, cmd: str) -> Tuple[int, str, str]:
        return self._adb_run(
            ["shell", cmd],
        )

    def pull(self, src: str, dst: str) -> Tuple[int, str, str]:
        return self._adb_run(["pull", src, dst])

    def push(self, src: str, dst: str) -> Tuple[int, str, str]:
        return self._adb_run(["push", src, dst])


class RVC4Benchmark(Benchmark):
    adb = AdbHandler()
    force_cpu: bool = False

    @property
    def default_configuration(self) -> Configuration:
        """
        profile: The SNPE profile to use for inference.
        runtime: The SNPE runtime to use for inference.
        num_images: The number of images to use for inference.
        dai_benchmark: Whether to use the DepthAI for benchmarking.
        repetitions: The number of repetitions to perform (dai-benchmark only).
        num_threads: The number of threads to use for inference (dai-benchmark only).
        num_messages: The number of messages to use for inference (dai-benchmark only).
        """
        return {
            "profile": "default",
            "runtime": "dsp",
            "num_images": 1000,
            "dai_benchmark": True,
            "repetitions": 10,
            "num_threads": 1,
            "num_messages": 50,
        }

    @property
    def all_configurations(self) -> List[Configuration]:
        return [{"profile": profile} for profile in PROFILES]

    def _get_input_sizes(self) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        csv_path = Path("info.csv")
        subprocess_run(
            [
                "snpe-dlc-info",
                "-i",
                self.model_path,
                "-s",
                csv_path,
            ]
        )
        content = csv_path.read_text()
        csv_path.unlink()

        start_marker = "Input Name,Dimensions,Type,Encoding Info"
        end_marker = "Output Name,Dimensions,Type,Encoding Info"
        start_index = content.find(start_marker)
        end_index = content.find(end_marker, start_index)

        # Extract and load the relevant CSV part into a pandas DataFrame.
        relevant_csv_part = content[start_index:end_index].strip()
        df = pd.read_csv(io.StringIO(relevant_csv_part))
        sizes = {
            str(row["Input Name"]): list(
                map(int, str(row["Dimensions"]).split(","))
            )
            for _, row in df.iterrows()
        }
        data_types = {
            str(row["Input Name"]): str(row["Type"])
            for _, row in df.iterrows()
        }

        return sizes, data_types

    def _prepare_raw_inputs(self, num_images: int) -> None:
        input_sizes, data_types = self._get_input_sizes()
        input_list = ""
        self.adb.shell(f"mkdir /data/local/tmp/{self.model_name}/inputs")
        for i in range(num_images):
            for name, size in input_sizes.items():
                if data_types[name] == "Float_32":
                    self.force_cpu = True
                    numpy_type = np.float32
                elif data_types[name] == "Float_16":
                    numpy_type = np.float16
                elif data_types[name] == "uFxp_8":
                    numpy_type = np.uint8
                else:
                    raise ValueError(
                        f"Unsupported data type {data_types[name]} for input {name}."
                    )
                img = cast(np.ndarray, np.random.rand(*size)).astype(
                    numpy_type
                )
                with tempfile.NamedTemporaryFile() as f:
                    img.tofile(f)
                    self.adb.push(
                        f.name,
                        f"/data/local/tmp/{self.model_name}/inputs/{name}_{i}.raw",
                    )

                input_list += f"{name}:=/data/local/tmp/{self.model_name}/inputs/{name}_{i}.raw "
            input_list += "\n"

        temp_path = tempfile.mktemp()
        with open(temp_path, "w") as f:
            f.write(input_list)
            f.flush()
        try:
            self.adb.push(
                temp_path, f"/data/local/tmp/{self.model_name}/input_list.txt"
            )
        finally:
            Path(temp_path).unlink()

    def _get_data_type(self) -> dai.TensorInfo.DataType:
        """Retrieve the data type of the model inputs. If the model is not a HubAI
        model, it defaults to dai.TensorInfo.DataType.U8F (INT8).

        @return: The data type of the model inputs.
        @rtype: dai.TensorInfo.DataType
        """
        from modelconverter.cli import Request, slug_to_id

        if not isinstance(
            self.model_path, str
        ) or not self.HUB_MODEL_PATTERN.match(self.model_path):
            return dai.TensorInfo.DataType.U8F

        model_id = slug_to_id(self.model_name, "models")
        model_variant = self.model_path.split(":")[1]

        model_variants = []
        for is_public in [True, False]:
            try:
                model_variants += Request.get(
                    "modelVersions/",
                    params={"model_id": model_id, "is_public": is_public},
                )
            except Exception:
                continue

        model_version_id = None
        for version in model_variants:
            if version["variant_slug"] == model_variant:
                model_version_id = version["id"]
                break

        if not model_version_id:
            return dai.TensorInfo.DataType.U8F

        model_instances = []
        for is_public in [True, False]:
            try:
                model_instances += Request.get(
                    "modelInstances/",
                    params={
                        "model_id": model_id,
                        "model_version_id": model_version_id,
                        "is_public": is_public,
                    },
                )
            except Exception:
                continue

        model_precision_type = "INT8"
        for instance in model_instances:
            if instance["platforms"] == ["RVC4"]:
                model_precision_type = instance.get(
                    "model_precision_type", "INT8"
                )
                break

        if model_precision_type == "FP16":
            return dai.TensorInfo.DataType.FP16
        elif model_precision_type == "FP32":
            self.force_cpu = True
            return dai.TensorInfo.DataType.FP32

        return dai.TensorInfo.DataType.U8F

    def benchmark(self, configuration: Configuration) -> BenchmarkResult:
        dai_benchmark = configuration.get("dai_benchmark")
        try:
            if dai_benchmark:
                for key in ["dai_benchmark", "num_images"]:
                    configuration.pop(key)
                return self._benchmark_dai(self.model_path, **configuration)
            else:
                for key in [
                    "dai_benchmark",
                    "repetitions",
                    "num_threads",
                    "num_messages",
                ]:
                    configuration.pop(key)
                return self._benchmark_snpe(self.model_path, **configuration)
        finally:
            if not dai_benchmark:
                # so we don't delete the wrong directory
                assert self.model_name

                self.adb.shell(f"rm -rf /data/local/tmp/{self.model_name}")

    def _benchmark_snpe(
        self,
        model_path: Path | str,
        num_images: int,
        profile: str,
        runtime: str,
    ) -> BenchmarkResult:
        runtime = RUNTIMES[runtime] if runtime in RUNTIMES else "use_dsp"

        if isinstance(model_path, str):
            model_archive = dai.getModelFromZoo(
                dai.NNModelDescription(
                    model_path,
                    platform=dai.Platform.RVC4.name,
                ),
                apiKey=environ.HUBAI_API_KEY if environ.HUBAI_API_KEY else "",
            )
            tmp_dir = Path(model_archive).parent / "tmp"
            shutil.unpack_archive(model_archive, tmp_dir)

            dlc_model_name = json.loads((tmp_dir / "config.json").read_text())[
                "model"
            ]["metadata"]["path"]
            dlc_path = next(tmp_dir.rglob(dlc_model_name), None)
            if not dlc_path:
                raise ValueError("Could not find model.dlc in the archive.")
            self.model_path = dlc_path
        elif str(model_path).endswith(".dlc"):
            dlc_path = model_path
        else:
            raise ValueError(
                "Unsupported model format. Supported formats: .dlc, or HubAI model slug."
            )

        self.adb.shell(f"mkdir /data/local/tmp/{self.model_name}")
        self.adb.push(
            str(dlc_path), f"/data/local/tmp/{self.model_name}/model.dlc"
        )
        self._prepare_raw_inputs(num_images)
        if self.force_cpu:
            logger.warning(
                "Forcing CPU runtime due to Float_32 input data type."
            )
            runtime = "use_cpu"

        _, stdout, _ = self.adb.shell(
            # "source /data/local/tmp/source_me.sh && "
            "snpe-parallel-run "
            f"--container /data/local/tmp/{self.model_name}/model.dlc "
            f"--input_list /data/local/tmp/{self.model_name}/input_list.txt "
            f"--output_dir /data/local/tmp/{self.model_name}/outputs "
            f"--perf_profile {profile} "
            "--cpu_fallback true "
            f"--{runtime}"
        )
        pattern = re.compile(r"(\d+\.\d+) infs/sec")
        match = pattern.search(stdout)
        if match is None:
            raise RuntimeError(
                "Could not find throughput in stdout. Likely a server error.\n"
                f"stdout:\n{stdout}"
            )
        fps = float(match.group(1))
        return BenchmarkResult(fps=fps, latency="N/A")

    def _benchmark_dai(
        self,
        model_path: Path | str,
        profile: str,
        runtime: str,
        repetitions: int,
        num_threads: int,
        num_messages: int,
    ) -> BenchmarkResult:
        device = dai.Device()

        if device.getPlatform() != dai.Platform.RVC4:
            raise ValueError(
                f"Found {device.getPlatformAsString()}, expected RVC4 platform."
            )

        if isinstance(model_path, str):
            modelPath = dai.getModelFromZoo(
                dai.NNModelDescription(
                    model_path,
                    platform=device.getPlatformAsString(),
                ),
                apiKey=environ.HUBAI_API_KEY if environ.HUBAI_API_KEY else "",
            )
        elif str(model_path).endswith(".tar.xz"):
            modelPath = str(model_path)
        elif str(model_path).endswith(".dlc"):
            raise ValueError(
                "DLC model format is not currently supported for dai-benchmark. Please use SNPE for DLC models."
            )
        else:
            raise ValueError(
                "Unsupported model format. Supported formats: .tar.xz, or HubAI model slug."
            )

        inputSizes = []
        inputNames = []
        if isinstance(model_path, str) or str(model_path).endswith(".tar.xz"):
            modelArhive = dai.NNArchive(modelPath)
            for input in modelArhive.getConfig().model.inputs:
                inputSizes.append(input.shape)
                inputNames.append(input.name)

        data_type = self._get_data_type()
        inputData = dai.NNData()
        for name, inputSize in zip(inputNames, inputSizes):
            img = np.random.randint(0, 255, inputSize, np.uint8)
            inputData.addTensor(name, img, dataType=data_type)

        with dai.Pipeline(device) as pipeline, Progress() as progress:
            repet_task = progress.add_task(
                "[magenta]Repetition", total=repetitions
            )

            benchmarkOut = pipeline.create(dai.node.BenchmarkOut)
            benchmarkOut.setRunOnHost(False)
            benchmarkOut.setFps(-1)

            neuralNetwork = pipeline.create(dai.node.NeuralNetwork)
            if isinstance(model_path, str) or str(model_path).endswith(
                ".tar.xz"
            ):
                neuralNetwork.setNNArchive(modelArhive)

            if self.force_cpu:
                logger.warning(
                    "Forcing CPU runtime due to Float_32 input data type."
                )
                runtime = "cpu"
            neuralNetwork.setBackendProperties(
                {
                    "runtime": runtime,
                    "performance_profile": profile,
                }
            )
            if num_threads > 1:
                logger.warning(
                    "num_threads > 1 is not supported for RVC4. Setting num_threads to 1."
                )
                num_threads = 1
            neuralNetwork.setNumInferenceThreads(num_threads)

            benchmarkIn = pipeline.create(dai.node.BenchmarkIn)
            benchmarkIn.setRunOnHost(False)
            benchmarkIn.sendReportEveryNMessages(num_messages)
            benchmarkIn.logReportsAsWarnings(False)

            benchmarkOut.out.link(neuralNetwork.input)
            neuralNetwork.out.link(benchmarkIn.input)

            outputQueue = benchmarkIn.report.createOutputQueue()
            inputQueue = benchmarkOut.input.createInputQueue()

            pipeline.start()
            inputQueue.send(inputData)

            rep = 0
            fps_list = []
            avg_latency_list = []
            while pipeline.isRunning() and rep < repetitions:
                benchmarkReport = outputQueue.get()
                if not isinstance(benchmarkReport, dai.BenchmarkReport):
                    raise ValueError(
                        f"Expected BenchmarkReport, got {type(benchmarkReport)}"
                    )
                fps = benchmarkReport.fps
                avg_latency = benchmarkReport.averageLatency * 1000

                fps_list.append(fps)
                avg_latency_list.append(avg_latency)
                progress.update(repet_task, advance=1)
                rep += 1

            # Currently, the latency measurement is only supported on RVC4 when using ImgFrame as the input to the BenchmarkOut which we don't do here.
            return BenchmarkResult(np.mean(fps_list), "N/A")
