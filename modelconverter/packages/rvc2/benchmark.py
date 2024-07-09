import logging
import time
from pathlib import Path
from typing import Dict, List, cast

import depthai as dai
import numpy as np
from depthai import NNData
from rich.progress import Progress

from ..base_benchmark import Benchmark, BenchmarkResult, Configuration

logger = logging.getLogger(__name__)


class RVC2Benchmark(Benchmark):
    @property
    def default_configuration(self) -> Configuration:
        """
        repetitions: The number of repetitions to perform.
        num_threads: The number of threads to use for inference.
        """
        return {"repetitions": 1, "num_threads": 2}

    @property
    def all_configurations(self) -> List[Configuration]:
        return [
            {"repetitions": 5, "num_threads": 1},
            {"repetitions": 5, "num_threads": 2},
            {"repetitions": 5, "num_threads": 3},
        ]

    def benchmark(self, configuration: Configuration) -> BenchmarkResult:
        return self._benchmark(self.model_path, **configuration)

    @staticmethod
    def _benchmark(
        model_path: Path, repetitions: int, num_threads: int
    ) -> BenchmarkResult:
        model = dai.OpenVINO.Blob(model_path)
        input_name_shape: Dict[str, List[int]] = {}
        input_name_type = {}
        for i in list(model.networkInputs):
            input_name_shape[i] = model.networkInputs[i].dims
            input_name_type[i] = model.networkInputs[i].dataType.name

        output_name_shape = {}
        output_name_type = {}
        for i in list(model.networkOutputs):
            output_name_shape[i] = model.networkOutputs[i].dims
            output_name_type[i] = model.networkOutputs[i].dataType.name

        pipeline = dai.Pipeline()

        detection_nn = pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(model_path)
        detection_nn.setNumInferenceThreads(num_threads)
        detection_nn.input.setBlocking(True)
        detection_nn.input.setQueueSize(1)

        nn_in = pipeline.createXLinkIn()
        nn_in.setMaxDataSize(6291456)
        nn_in.setStreamName("in_nn")
        nn_in.out.link(detection_nn.input)

        xout_nn = pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")
        xout_nn.input.setQueueSize(1)
        xout_nn.input.setBlocking(True)
        detection_nn.out.link(xout_nn.input)

        xlink_buffer_max_size = 5 * 1024 * 1024
        product_sum = sum(
            map(lambda x: np.product(np.array(x)), output_name_shape.values())
        )

        xlink_buffer_count = int(xlink_buffer_max_size / product_sum)

        logger.info(f"XLink buffer count: {xlink_buffer_count}")
        if xlink_buffer_count > 1000:
            logger.warning(
                "XLink buffer count is too high! "
                "The benchmarking will take more time and "
                "the results may be overestimated."
            )

        with dai.Device(pipeline) as device, Progress() as progress:
            device = cast(dai.Device, device)
            detection_in_count = 100 + xlink_buffer_count
            detection_in = device.getInputQueue(
                "in_nn", maxSize=detection_in_count, blocking=True
            )
            q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=True)

            fps_storage = []
            diffs = []
            time.sleep(1)
            repet_task = progress.add_task(
                "[magenta]Repetition", total=repetitions
            )
            infer_task = progress.add_task(
                "[magenta]Inference", total=300 + 2 * xlink_buffer_count
            )
            for _ in range(repetitions):
                progress.reset(infer_task, total=300 + 2 * xlink_buffer_count)
                for _ in range(100 + xlink_buffer_count):
                    nn_data = None
                    for inp_name in input_name_shape:
                        if input_name_type[inp_name] in ["FLOAT16", "FLOAT32"]:
                            frame = cast(
                                np.ndarray,
                                np.random.rand(*input_name_shape[inp_name]),
                            )
                            frame = frame.astype(
                                "float16"
                                if input_name_type[inp_name] == "FLOAT16"
                                else "float32"
                            )
                        elif input_name_type[inp_name] in ["INT", "I8", "U8F"]:
                            frame = np.random.randint(
                                256,
                                size=input_name_shape[inp_name],
                                dtype=np.int32
                                if input_name_type[inp_name] == "INT"
                                else np.uint8
                                if input_name_type[inp_name] == "U8F"
                                else np.int8,
                            )
                        else:
                            raise RuntimeError(
                                f"Unknown input type detected: {input_name_type[inp_name]}!"
                            )

                        nn_data = dai.NNData()
                        nn_data.setLayer(inp_name, frame)

                    if nn_data is None:
                        raise RuntimeError("No input data was created!")
                    detection_in.send(nn_data)
                    progress.update(infer_task, advance=1)

                for _ in range(100):
                    progress.update(infer_task, advance=1)
                    time.sleep(3 / 100)

                for _ in range(40 + xlink_buffer_count):
                    cast(NNData, q_nn.get()).getFirstLayerFp16()
                    progress.update(infer_task, advance=1)

                start = time.time()
                for _ in range(50):
                    cast(NNData, q_nn.get()).getFirstLayerFp16()
                    progress.update(infer_task, advance=1)
                diff = time.time() - start
                diffs.append(diff / 50)
                fps_storage.append(50 / diff)

                for _ in range(10):
                    cast(NNData, q_nn.get()).getFirstLayerFp16()
                    progress.update(infer_task, advance=1)
                progress.update(repet_task, advance=1)

            diffs = np.array(diffs) * 1000
            return BenchmarkResult(np.mean(fps_storage), np.mean(diffs))
