import io
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, cast

import depthai as dai
import numpy as np
import pandas as pd
from rich.progress import Progress

from modelconverter.utils import subprocess_run

from ..base_benchmark import Benchmark, BenchmarkResult, Configuration

logger = logging.getLogger(__name__)

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

    def _get_input_sizes(self) -> Dict[str, List[int]]:
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
        end_marker = "Total parameters:"
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
        return sizes

    def _prepare_raw_inputs(self, num_images: int) -> None:
        input_sizes = self._get_input_sizes()
        input_list = ""
        self.adb.shell(f"mkdir /data/local/tmp/{self.model_name}/inputs")
        for i in range(num_images):
            for name, size in input_sizes.items():
                img = cast(np.ndarray, np.random.rand(*size)).astype(
                    np.float32
                )
                with tempfile.TemporaryFile() as f:
                    img.tofile(f)
                    self.adb.push(
                        f.name,
                        f"/data/local/tmp/{self.model_name}/inputs/{name}_{i}.raw",
                    )

                input_list += f"{name}:=/data/local/tmp/{self.model_name}/inputs/{name}_{i}.raw "
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(input_list)
            self.adb.push(
                f.name, f"/data/local/tmp/{self.model_name}/input_list.txt"
            )

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
        self.adb.shell(f"mkdir /data/local/tmp/{self.model_name}")
        self.adb.push(
            str(model_path), f"/data/local/tmp/{self.model_name}/model.dlc"
        )
        self._prepare_raw_inputs(num_images)

        _, stdout, _ = self.adb.shell(
            "source /data/local/tmp/source_me.sh && "
            "snpe-parallel-run "
            f"--container /data/local/tmp/{self.model_name}/model.dlc "
            f"--input_list /data/local/tmp/{self.model_name}/input_list.txt "
            f"--output_dir /data/local/tmp/{self.model_name}/outputs "
            f"--perf_profile {profile} "
            "--cpu_fallback false "
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
        return BenchmarkResult(fps=fps, latency=0)

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
                )
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

        inputData = dai.NNData()
        for name, inputSize in zip(inputNames, inputSizes):
            img = np.random.randint(
                0, 255, (1, inputSize[1], inputSize[2], 3), np.uint8
            )
            inputData.addTensor(name, img)

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
