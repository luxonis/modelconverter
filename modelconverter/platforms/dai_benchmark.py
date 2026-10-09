"""The DepthAI benchmark pipeline of the RVC2 and RVC4 benchmarks.

Benchmark nodes on the device feed one input to the model in a loop and
report the throughput. The latency of one inference comes from the
device logs.
"""

import re
from collections.abc import Callable

import depthai as dai
import numpy as np

from modelconverter.platforms.base_benchmark import Result
from modelconverter.utils.log_latency import parse_inference_latency
from modelconverter.utils.progress_handler import create_progress_handler


def run_dai_benchmark(
    device: dai.Device,
    input_data: dai.NNData,
    configure_network: Callable[[dai.node.NeuralNetwork], None],
    *,
    latency_pattern: re.Pattern[str],
    log_level: dai.LogLevel,
    input_fps: float,
    num_threads: int,
    num_messages: int,
    benchmark_time: int,
    repetitions: int,
) -> Result:
    """Run the model on the device in a loop and measure its speed.

    Args:
        device: Device to run the pipeline on.
        input_data: Input that is sent to the model on every inference.
        configure_network: Loads the model into the network node.
        latency_pattern: Pattern of the latency in the device logs,
            with a ``latency_ms`` group.
        log_level: Log level at which the device reports the latency.
            The logs reach the callback only; stdout gets warnings.
        input_fps: Rate of the inputs, ``-1`` for no limit.
        num_threads: Number of inference threads of the network.
        num_messages: Number of messages in each throughput report.
        benchmark_time: Duration of the run in seconds. A value that is
            not positive runs ``repetitions`` reports instead.
        repetitions: Number of reports to collect in a run that
            ``benchmark_time`` does not limit.

    Returns:
        The mean ``fps`` and the mean ``latency`` in milliseconds, or
        ``"N/A"`` when the logs hold no latency.
    """
    latencies: list[float] = []

    def on_log_message(log_message: dai.LogMessage) -> None:
        """Collect the inference latency from a device log message."""
        latency = parse_inference_latency(log_message, latency_pattern)
        if latency is not None:
            latencies.append(latency)

    callback_id = device.addLogCallback(on_log_message)
    device.setLogLevel(log_level)
    device.setLogOutputLevel(dai.LogLevel.WARN)

    fps_list = []
    try:
        with dai.Pipeline(device) as pipeline:
            benchmark_out = pipeline.create(dai.node.BenchmarkOut)
            benchmark_out.setRunOnHost(False)
            benchmark_out.setFps(input_fps)

            neural_network = pipeline.create(dai.node.NeuralNetwork)
            configure_network(neural_network)
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
            input_queue.send(input_data)

            progress, on_tick, should_continue = create_progress_handler(
                benchmark_time, repetitions
            )

            with progress:
                while pipeline.isRunning() and should_continue():
                    benchmark_report = output_queue.get()
                    if not isinstance(benchmark_report, dai.BenchmarkReport):
                        raise TypeError(
                            "Expected BenchmarkReport, got "
                            f"{type(benchmark_report)}"
                        )

                    fps_list.append(benchmark_report.fps)
                    on_tick()
    finally:
        device.removeLogCallback(callback_id)

    return {
        "fps": float(np.mean(fps_list)),
        "latency": float(np.mean(latencies)) if latencies else "N/A",
    }
