"""Benchmarking of converted models on RVC3 devices.

Loads the model through the OpenVINO inference engine onto the ``VPUX``
plugin of a connected RVC3 device and measures its throughput and
latency under a varying number of concurrent inference requests.
"""

import sys
from datetime import datetime, timezone
from statistics import median

from loguru import logger
from openvino.inference_engine.ie_api import IECore, StatusCode
from rich.progress import track

from modelconverter.platforms.base_benchmark import (
    Benchmark,
    Configuration,
    Result,
    get_option,
)


class RVC3Benchmark(Benchmark):
    """Benchmark of a converted model running on an RVC3 device."""

    @property
    def default_configuration(self) -> Configuration:
        """Default configuration for RVC3 benchmarking.

        Options:
            requests: The number of requests to perform.
        """
        return {"requests": 1}

    @property
    def all_configurations(self) -> list[Configuration]:
        """Return the configurations used by the full benchmark.

        Covers one to five concurrent inference requests.
        """
        return [{"requests": i} for i in range(1, 6)]

    def benchmark(self, configuration: Configuration) -> Result:
        """Run a single benchmark of the model on the device.

        Args:
            configuration: Configuration to benchmark with. The only
                supported option is ``requests``, the number of
                concurrent inference requests.

        Returns:
            The measured throughput under ``fps`` and the median
            request latency under ``latency``.

        """
        return self._benchmark(
            str(self.model_path),
            requests=get_option(configuration, "requests", int),
        )

    @staticmethod
    def _benchmark(model_path: str, requests: int) -> Result:
        ie = IECore()
        exe_network = ie.load_network(
            model_path,
            "VPUX",
            config={},
            num_requests=requests,
        )
        requests = len(exe_network.requests)
        infer_requests = exe_network.requests

        times = []
        in_fly = set()
        iterations = 1000
        i = 0
        start_time = datetime.now(timezone.utc)
        for _ in track(
            range(max(requests, iterations)),
            description="Running inference",
        ):
            infer_request_id = exe_network.get_idle_request_id()
            if infer_request_id < 0:
                status = exe_network.wait(num_requests=1)
                if status != StatusCode.OK:
                    raise RuntimeError("Wait for idle request failed!")
                infer_request_id = exe_network.get_idle_request_id()
                if infer_request_id < 0:
                    raise RuntimeError("Invalid request id!")
            if infer_request_id in in_fly:
                times.append(infer_requests[infer_request_id].latency)
            else:
                in_fly.add(infer_request_id)
            infer_requests[infer_request_id].async_infer()
            i += 1

        status = exe_network.wait()
        if status != StatusCode.OK:
            logger.critical(
                f"Wait for all requests has failed with status code {status}!"
            )
            sys.exit(1)

        total_duration_sec = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()
        times.extend(
            infer_requests[infer_request_id].latency
            for infer_request_id in in_fly
        )
        times.sort()
        latency = median(times)
        fps = i / total_duration_sec
        return {"fps": fps, "latency": latency}
