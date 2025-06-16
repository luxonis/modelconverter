import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeAlias

import polars as pl
from loguru import logger

from modelconverter.utils import is_hubai_available, resolve_path


class BenchmarkResult(NamedTuple):
    """Benchmark result, tuple (FPS, latency in ms)"""

    fps: float
    latency: float | Literal["N/A"]


Configuration: TypeAlias = dict[str, Any]


class Benchmark(ABC):
    VALID_EXTENSIONS = (".tar.xz", ".blob", ".dlc")
    HUB_MODEL_PATTERN = re.compile(r"^(?:([^/]+)/)?([^:]+):([^:]+)(?::(.+))?$")

    def __init__(
        self,
        model_path: str,
        dataset_path: Path | None = None,
    ):
        if any(model_path.endswith(ext) for ext in self.VALID_EXTENSIONS):
            self.model_path = resolve_path(model_path, Path.cwd())
            self.model_name = self.model_path.stem
            self.model_instance = None
        else:
            hub_match = self.HUB_MODEL_PATTERN.match(model_path)
            if not hub_match:
                raise ValueError(
                    "Invalid 'model-path' format. Expected either:\n"
                    "- Model file path: path/to/model.blob, path/to/model.dlc or path/to/model.tar.xz\n"
                    "- HubAI model slug: [team_name/]model_name:variant[:model_instance]"
                )
            (
                team_name,
                model_name,
                model_variant,
                model_instance,
            ) = hub_match.groups()
            if is_hubai_available(model_name, model_variant):
                self.model_path = model_path
                self.model_name = model_name
                self.model_instance = model_instance
            else:
                raise ValueError(
                    f"Model {team_name + '/' if team_name else ''}{model_name}:{model_variant}{':' + model_instance if model_instance else ''} not found in HubAI."
                )

        self.dataset_path = dataset_path

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
    def all_configurations(self) -> list[Configuration]:
        pass

    def print_results(
        self, results: list[tuple[Configuration, BenchmarkResult]]
    ) -> None:
        assert results, "No results to print"

        from rich import box
        from rich.console import Console
        from rich.table import Table

        table = Table(
            title=f"Benchmark Results for [yellow]{self.model_name}",
            box=box.ROUNDED,
        )

        updated_header = [
            *results[0][0].keys(),
            "fps",
            "latency (ms)",
        ]
        for field in updated_header:
            table.add_column(f"[cyan]{field}")
        for configuration, result in results:
            fps_color = (
                "yellow"
                if 5 < result.fps < 15
                else "red"
                if result.fps < 5
                else "green"
            )
            if isinstance(result.latency, str):
                latency_color = "orange3"
            else:
                latency_color = (
                    "yellow"
                    if 50 < result.latency < 100
                    else "red"
                    if result.latency > 100
                    else "green"
                )
            table.add_row(
                *(f"[magenta]{x}" for x in configuration.values()),
                f"[{fps_color}]{result.fps:.2f}",
                f"[{latency_color}]{result.latency}"
                if isinstance(result.latency, str)
                else f"[{latency_color}]{result.latency:.5f}",
            )
        console = Console()
        console.print(table)

    def save_results(
        self, results: list[tuple[Configuration, BenchmarkResult]]
    ) -> None:
        assert results, "No results to save"
        df = pl.DataFrame(
            [
                {**configuration, **result._asdict()}
                for configuration, result in results
            ]
        )
        file = f"{self.model_name}_benchmark_results.csv"
        df.write_csv(file)
        logger.info(f"Benchmark results saved to {file}.")

    def run(self, full: bool = True, save: bool = False, **kwargs) -> None:
        logger.info(f"Running benchmarking for {self.model_name}")
        for key, value in self.default_configuration.items():
            if key in kwargs and value is not None:
                kwargs[key] = type(value)(kwargs[key])

        configurations = (
            [{**self.default_configuration, **kwargs}]
            if not full
            else self.all_configurations
        )
        results: list[tuple[Configuration, BenchmarkResult]] = [
            (configuration, self.benchmark(configuration))
            for configuration in configurations
        ]
        self.print_results(results)
        if save:
            self.save_results(results)
