"""Benchmarking of converted models on an RVC2 device.

Runs a ``.blob`` or an NN Archive on a connected RVC2 device through a
DepthAI benchmark pipeline, feeding it random data of the shape the
model declares, and reports the throughput and the inference latency the
device logs.
"""

from pathlib import Path

import depthai as dai
import numpy as np
from luxonis_ml.typing import PathType

from modelconverter.platforms.base_benchmark import (
    Benchmark,
    Configuration,
    Result,
    get_input_fps,
    get_option,
)
from modelconverter.utils import environ
from modelconverter.utils.log_latency import (
    RVC2_INFERENCE_LATENCY_RE,
    run_dai_benchmark,
)


class RVC2Benchmark(Benchmark):
    """Benchmark of a model running on a connected RVC2 device."""

    @property
    def default_configuration(self) -> Configuration:
        """Default configuration for RVC2 benchmarking.

        Configuration options:
            ``repetitions``
                The number of repetitions to perform (ignored if
                ``benchmark_time`` is set).
            ``benchmark_time``
                Duration in seconds for time-based benchmarking (overrides
                ``repetitions``).
            ``input_fps``
                Rate at which inputs are sent. ``-1`` removes the
                limit.
            ``num_messages``
                The number of messages measured for each report.
            ``num_threads``
                The number of threads to use for inference.
        """
        return {
            "repetitions": 10,
            "benchmark_time": 20,
            "input_fps": -1.0,
            "num_messages": 50,
            "num_threads": 2,
        }

    @property
    def all_configurations(self) -> list[Configuration]:
        """Return the configurations used by the full benchmark.

        Covers one, two and three inference threads, leaving the
        remaining options to the caller.
        """
        return [{"num_threads": i} for i in [1, 2, 3]]

    def benchmark(self, configuration: Configuration) -> Result:
        """Run a single benchmark of the model on the device.

        Args:
            configuration: Configuration to benchmark with. All the
                options of `default_configuration` must be present.

        Returns:
            The mean throughput under ``fps`` and the mean inference
            latency in milliseconds under ``latency``, the latter being
            ``"N/A"`` when the device logged no latency at all.

        """
        return self._benchmark(
            self.model_path,
            repetitions=get_option(configuration, "repetitions", int),
            num_messages=get_option(configuration, "num_messages", int),
            num_threads=get_option(configuration, "num_threads", int),
            benchmark_time=get_option(configuration, "benchmark_time", int),
            input_fps=get_input_fps(configuration),
        )

    @staticmethod
    def _benchmark(
        model_path: PathType,
        repetitions: int,
        num_messages: int,
        num_threads: int,
        benchmark_time: int,
        input_fps: float,
    ) -> Result:
        device = dai.Device()
        if device.getPlatform() != dai.Platform.RVC2:
            raise ValueError(
                f"Found {device.getPlatformAsString()}, expected RVC2 platform."
            )
        model = _load_model(model_path, device)

        def configure_network(network: dai.node.NeuralNetwork) -> None:
            """Load the archive or the blob into the network node."""
            if isinstance(model, dai.NNArchive):
                network.setNNArchive(model)
            else:
                network.setBlobPath(model)

        # RVC2 reports per-inference latency at TRACE level.
        return run_dai_benchmark(
            device,
            _random_input_data(model),
            configure_network,
            latency_pattern=RVC2_INFERENCE_LATENCY_RE,
            log_level=dai.LogLevel.TRACE,
            input_fps=input_fps,
            num_threads=num_threads,
            num_messages=num_messages,
            benchmark_time=benchmark_time,
            repetitions=repetitions,
        )


def _load_model(
    model_path: PathType, device: dai.Device
) -> dai.NNArchive | Path:
    """Load the NN Archive of a path or a HubAI slug, or keep a blob path."""
    if isinstance(model_path, str):
        return dai.NNArchive(
            Path(
                dai.getModelFromZoo(
                    dai.NNModelDescription(
                        model_path, platform=device.getPlatformAsString()
                    ),
                    apiKey=environ.HUBAI_API_KEY or "",
                )
            )
        )
    if str(model_path).endswith(".tar.xz"):
        return dai.NNArchive(model_path)
    if model_path.suffix == ".blob":
        return model_path
    raise ValueError(
        "Unsupported model format. Supported formats: .tar.xz, .blob, or HubAI model slug."
    )


def _random_input_data(model: dai.NNArchive | Path) -> dai.NNData:
    """Build a random 8-bit HWC image for each input of the model."""
    if isinstance(model, dai.NNArchive):
        sizes = [
            (archive_input.name, archive_input.shape[::-1])
            for archive_input in model.getConfig().model.inputs
        ]
    else:
        blob = dai.OpenVINO.Blob(model)
        sizes = [
            (name, blob.networkInputs[name].dims)
            for name in blob.networkInputs
        ]

    input_data = dai.NNData()
    for name, size in sizes:
        img = np.random.randint(0, 255, (size[1], size[0], 3), np.uint8)
        input_data.addTensor(name, img)
    return input_data
