import time
from pathlib import Path

import depthai as dai
import numpy as np
from rich.progress import Progress, TextColumn

from modelconverter.packages.base_benchmark import (
    Benchmark,
    BenchmarkResult,
    Configuration,
)
from modelconverter.utils import environ


class RVC2Benchmark(Benchmark):
    @property
    def default_configuration(self) -> Configuration:
        """
        repetitions: The number of repetitions to perform (ignored if benchmark_time is set).
        benchmark_time: Duration in seconds for time-based benchmarking (overrides repetitions).
        num_messages: The number of messages to send for benchmarking.
        num_threads: The number of threads to use for inference.
        """
        return {
            "repetitions": 10,
            "benchmark_time": None,
            "num_messages": 50,
            "num_threads": 2,
        }

    @property
    def all_configurations(self) -> list[Configuration]:
        return [{"num_threads": i} for i in [1, 2, 3]]

    def benchmark(self, configuration: Configuration) -> BenchmarkResult:
        return self._benchmark(self.model_path, **configuration)

    @staticmethod
    def _benchmark(
        model_path: str | Path,
        repetitions: int,
        num_messages: int,
        num_threads: int,
        benchmark_time: int | None = None,
    ) -> BenchmarkResult:
        device = dai.Device()
        if device.getPlatform() != dai.Platform.RVC2:
            raise ValueError(
                f"Found {device.getPlatformAsString()}, expected RVC2 platform."
            )

        if isinstance(model_path, str):
            modelPath = Path(
                dai.getModelFromZoo(
                    dai.NNModelDescription(
                        model_path,
                        platform=device.getPlatformAsString(),
                    ),
                    apiKey=environ.HUBAI_API_KEY
                    if environ.HUBAI_API_KEY
                    else "",
                )
            )
        elif (
            str(model_path).endswith(".tar.xz") or model_path.suffix == ".blob"
        ):
            modelPath = model_path
        else:
            raise ValueError(
                "Unsupported model format. Supported formats: .tar.xz, .blob, or HubAI model slug."
            )

        inputSizes = []
        inputNames = []
        if isinstance(model_path, str) or str(model_path).endswith(".tar.xz"):
            modelArhive = dai.NNArchive(str(modelPath))
            for input in modelArhive.getConfig().model.inputs:
                inputSizes.append(input.shape[::-1])
                inputNames.append(input.name)
        elif str(model_path).endswith(".blob"):
            blob_model = dai.OpenVINO.Blob(modelPath)
            for input in blob_model.networkInputs:
                inputSizes.append(blob_model.networkInputs[input].dims)
                inputNames.append(input)

        inputData = dai.NNData()
        for name, inputSize in zip(inputNames, inputSizes, strict=True):
            img = np.random.randint(
                0, 255, (inputSize[1], inputSize[0], 3), np.uint8
            )
            inputData.addTensor(name, img)

        with dai.Pipeline(device) as pipeline:
            if benchmark_time:

                def format_time(seconds: float) -> str:
                    mins = int(seconds // 60)
                    secs = int(seconds % 60)
                    return f"{mins:02d}:{secs:02d}"

                progress = Progress(TextColumn("{task.description}"))
                progress_task = progress.add_task("", total=benchmark_time)
            else:
                progress = Progress()
                progress_task = progress.add_task(
                    "[magenta]Repetition", total=repetitions
                )

            progress.start()

            benchmarkOut = pipeline.create(dai.node.BenchmarkOut)
            benchmarkOut.setRunOnHost(False)
            benchmarkOut.setFps(-1)

            neuralNetwork = pipeline.create(dai.node.NeuralNetwork)
            if isinstance(model_path, str) or str(model_path).endswith(
                ".tar.xz"
            ):
                neuralNetwork.setNNArchive(modelArhive)
            elif str(model_path).endswith(".blob"):
                neuralNetwork.setBlobPath(modelPath)
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

            if benchmark_time:
                start_time = time.time()
                while (
                    pipeline.isRunning()
                    and (time.time() - start_time) < benchmark_time
                ):
                    benchmarkReport = outputQueue.get()
                    if not isinstance(benchmarkReport, dai.BenchmarkReport):
                        raise TypeError(
                            f"Expected BenchmarkReport, got {type(benchmarkReport)}"
                        )
                    fps = benchmarkReport.fps
                    avg_latency = benchmarkReport.averageLatency * 1000

                    fps_list.append(fps)
                    avg_latency_list.append(avg_latency)

                    elapsed = time.time() - start_time
                    capped_elapsed = min(elapsed, benchmark_time)
                    elapsed_str = format_time(capped_elapsed)
                    total_str = format_time(benchmark_time)
                    progress.update(
                        progress_task,
                        description=f"[magenta]Time Elapsed (mm:ss) [cyan]{elapsed_str} / {total_str}",
                    )
            else:
                while pipeline.isRunning() and rep < repetitions:
                    benchmarkReport = outputQueue.get()
                    if not isinstance(benchmarkReport, dai.BenchmarkReport):
                        raise TypeError(
                            f"Expected BenchmarkReport, got {type(benchmarkReport)}"
                        )
                    fps = benchmarkReport.fps
                    avg_latency = benchmarkReport.averageLatency * 1000

                    fps_list.append(fps)
                    avg_latency_list.append(avg_latency)
                    progress.update(progress_task, advance=1)
                    rep += 1
            progress.stop()

            # Currently, the latency measurement is not supported on RVC2 by the depthai library.
            return BenchmarkResult(float(np.mean(fps_list)), "N/A")
