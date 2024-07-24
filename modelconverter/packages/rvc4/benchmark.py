import io
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

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
        num_images: The number of images to use for inference.
        """
        return {"profile": "default", "num_images": 1000}

    @property
    def all_configurations(self) -> List[Configuration]:
        return [{"profile": profile} for profile in PROFILES]

    def _get_input_sizes(self) -> Dict[str, List[int]]:
        csv_path = Path("info.csv")
        subprocess_run(
            ["snpe-dlc-info", "-i", self.model_path, "-s", csv_path]
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
        try:
            return self._benchmark(self.model_path, **configuration)
        finally:
            # so we don't delete the wrong directory
            assert self.model_name

            self.adb.shell(f"rm -rf /data/local/tmp/{self.model_name}")

    def _benchmark(
        self, model_path: Path, num_images: int, profile: str
    ) -> BenchmarkResult:
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
            "--use_dsp"
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
