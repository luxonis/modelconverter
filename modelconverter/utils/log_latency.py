"""The DepthAI benchmark pipeline and its latency from device logs.

The RVC2 and RVC4 benchmarks run the same pipeline: benchmark nodes on
the device feed one input to the model in a loop and report the
throughput. They measure how long a single inference takes by reading
the log messages the device emits through ``depthai``. The two
platforms word those messages differently, so a regular expression is
provided for each.
"""

import re
from collections.abc import Callable

import depthai as dai
import numpy as np

from modelconverter.utils.progress_handler import create_progress_handler

RVC2_INFERENCE_LATENCY_RE = re.compile(
    r"NeuralNetwork inference took\s+'(?P<latency_ms>[0-9]+(?:\.[0-9]+)?)'\s+ms"
)
RVC4_INFERENCE_LATENCY_RE = re.compile(
    r"Executing the model took:\s*(?P<latency_ms>[0-9]+(?:\.[0-9]+)?)\s*ms"
)


def parse_inference_latency(
    log_message: dai.LogMessage, pattern: re.Pattern[str]
) -> float | None:
    """Extract an inference duration in milliseconds from a device log
    message.

    Args:
        log_message: Log message received from the device.
        pattern: Regular expression to search the message payload
            with. Must define a ``latency_ms`` group.

    Returns:
        The latency in milliseconds, or ``None`` if the payload does
        not match the pattern.

    """
    payload = log_message.payload
    if isinstance(payload, bytes):
        payload = payload.decode(errors="replace")

    match = pattern.search(str(payload))
    return float(match.group("latency_ms")) if match else None


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
) -> dict[str, float | str | None]:
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
