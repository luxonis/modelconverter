import logging
from datetime import datetime
from statistics import median
from typing import List

from openvino.inference_engine.ie_api import IECore, StatusCode
from rich.progress import track

from ..base_benchmark import Benchmark, BenchmarkResult, Configuration

logger = logging.getLogger(__name__)


class RVC3Benchmark(Benchmark):
    @property
    def default_configuration(self) -> Configuration:
        """
        requests: The number of requests to perform.
        """
        return {"requests": 1}

    @property
    def all_configurations(self) -> List[Configuration]:
        return [{"requests": i} for i in range(1, 6)]

    def benchmark(self, configuration: Configuration) -> BenchmarkResult:
        return self._benchmark(str(self.model_path), **configuration)

    @staticmethod
    def _benchmark(model_path: str, requests: int) -> BenchmarkResult:
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
        start_time = datetime.utcnow()
        for _ in track(
            range(iterations if iterations > requests else requests),
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
            exit(1)

        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        for infer_request_id in in_fly:
            times.append(infer_requests[infer_request_id].latency)
        times.sort()
        latency = median(times)
        fps = i / total_duration_sec
        return BenchmarkResult(fps, latency)
