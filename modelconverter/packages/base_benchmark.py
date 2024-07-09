from abc import ABC, abstractmethod
from collections import namedtuple
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from typing_extensions import TypeAlias

from modelconverter.utils import resolve_path

logger = getLogger(__name__)


BenchmarkResult = namedtuple("Result", ["fps", "latency"])
"""Benchmark result, tuple (FPS, latency in ms)"""

Configuration: TypeAlias = Dict[str, Any]
"""Configuration dictionary, package specific.

i.e. `{"shaves": 4}` for RVC2
"""


class Benchmark(ABC):
    def __init__(
        self,
        model_path: str,
        dataset_path: Optional[Path] = None,
    ):
        self.model_path = resolve_path(model_path, Path.cwd())
        self.dataset_path = dataset_path
        self.model_name = self.model_path.stem
        self.header = [
            *self.default_configuration.keys(),
            "fps",
            "latency (ms)",
        ]

    @abstractmethod
    def benchmark(self, configuration: Configuration) -> BenchmarkResult:
        pass

    @property
    @abstractmethod
    def default_configuration(self) -> Configuration:
        pass

    @property
    @abstractmethod
    def all_configurations(self) -> List[Configuration]:
        pass

    def print_results(
        self, results: List[Tuple[Configuration, BenchmarkResult]]
    ) -> None:
        assert results, "No results to print"

        from rich import box
        from rich.console import Console
        from rich.table import Table

        table = Table(
            title=f"Benchmark Results for [yellow]{self.model_name}",
            box=box.ROUNDED,
        )
        for field in self.header:
            table.add_column(f"[cyan]{field}")
        for configuration, result in results:
            fps_color = (
                "yellow"
                if 5 < result.fps < 15
                else "red"
                if result.fps < 5
                else "green"
            )
            latency_color = (
                "yellow"
                if 50 < result.latency < 100
                else "red"
                if result.latency > 100
                else "green"
            )
            table.add_row(
                *map(lambda x: f"[magenta]{x}", configuration.values()),
                f"[{fps_color}]{result.fps:.2f}",
                f"[{latency_color}]{result.latency:.5f}",
            )
        console = Console()
        console.print(table)

    def save_results(
        self, results: List[Tuple[Configuration, BenchmarkResult]]
    ) -> None:
        assert results, "No results to save"
        df = pd.DataFrame(
            [
                {**configuration, **result._asdict()}
                for configuration, result in results
            ]
        )
        file = f"{self.model_name}_benchmark_results.csv"
        df.to_csv(file, index=False)
        logger.info(f"Benchmark results saved to {file}.")

    def run(self, full: bool = True, save: bool = False, **kwargs) -> None:
        logger.info(f"Running benchmarking for {self.model_name}")
        for key, value in self.default_configuration.items():
            if key in kwargs:
                kwargs[key] = type(value)(kwargs[key])

        configurations = (
            [{**self.default_configuration, **kwargs}]
            if not full
            else self.all_configurations
        )
        results: List[Tuple[Configuration, BenchmarkResult]] = [
            (configuration, self.benchmark(configuration))
            for configuration in configurations
        ]
        self.print_results(results)
        if save:
            self.save_results(results)
