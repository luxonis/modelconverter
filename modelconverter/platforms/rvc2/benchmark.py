from pathlib import Path

import depthai as dai
import numpy as np
from luxonis_ml.typing import PathType

from modelconverter.platforms.base_benchmark import (
    Benchmark,
    Configuration,
    Result,
    get_option,
)
from modelconverter.utils import create_progress_handler, environ
from modelconverter.utils.log_latency import (
    RVC2_INFERENCE_LATENCY_RE,
    parse_inference_latency,
)


class RVC2Benchmark(Benchmark):
    @property
    def default_configuration(self) -> Configuration:
        """Default configuration for RVC2 benchmarking.

        Options:
            repetitions: The number of repetitions to perform (ignored if
            benchmark_time is set).

            benchmark_time: Duration in seconds for time-based benchmarking (overrides repetitions).
            num_messages: The number of messages to send for benchmarking.
            num_threads: The number of threads to use for inference.
        """
        return {
            "repetitions": 10,
            "benchmark_time": 20,
            "num_messages": 50,
            "num_threads": 2,
        }

    @property
    def all_configurations(self) -> list[Configuration]:
        return [{"num_threads": i} for i in [1, 2, 3]]

    def benchmark(self, configuration: Configuration) -> Result:
        return self._benchmark(
            self.model_path,
            repetitions=get_option(configuration, "repetitions", int),
            num_messages=get_option(configuration, "num_messages", int),
            num_threads=get_option(configuration, "num_threads", int),
            benchmark_time=get_option(configuration, "benchmark_time", int),
        )

    @staticmethod
    def _benchmark(
        model_path: PathType,
        repetitions: int,
        num_messages: int,
        num_threads: int,
        benchmark_time: int,
    ) -> Result:
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
                    apiKey=environ.HUBAI_API_KEY or "",
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
            modelArchive = dai.NNArchive(str(modelPath))  # type: ignore[arg-type]
            for input in modelArchive.getConfig().model.inputs:
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

        latencies: list[float] = []

        def on_log_message(log_message: dai.LogMessage) -> None:
            latency = parse_inference_latency(
                log_message, RVC2_INFERENCE_LATENCY_RE
            )
            if latency is not None:
                latencies.append(latency)

        callback_id = device.addLogCallback(on_log_message)
        # RVC2 reports per-inference latency at TRACE level.  Keep TRACE logs
        # out of stdout; the callback still receives the messages.
        device.setLogLevel(dai.LogLevel.TRACE)
        device.setLogOutputLevel(dai.LogLevel.WARN)

        fps_list = []
        try:
            with dai.Pipeline(device) as pipeline:
                benchmarkOut = pipeline.create(dai.node.BenchmarkOut)
                benchmarkOut.setRunOnHost(False)
                benchmarkOut.setFps(-1)

                neuralNetwork = pipeline.create(dai.node.NeuralNetwork)
                if isinstance(model_path, str) or str(model_path).endswith(
                    ".tar.xz"
                ):
                    neuralNetwork.setNNArchive(modelArchive)
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

                progress, on_tick, should_continue = create_progress_handler(
                    benchmark_time, repetitions
                )

                with progress:
                    while pipeline.isRunning() and should_continue():
                        benchmarkReport = outputQueue.get()
                        if not isinstance(
                            benchmarkReport, dai.BenchmarkReport
                        ):
                            raise TypeError(
                                f"Expected BenchmarkReport, got {type(benchmarkReport)}"
                            )

                        fps_list.append(benchmarkReport.fps)
                        on_tick()
        finally:
            device.removeLogCallback(callback_id)

        return {
            "fps": float(np.mean(fps_list)),
            "latency": float(np.mean(latencies)) if latencies else "N/A",
        }
