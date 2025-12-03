import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeAlias

import polars as pl
from loguru import logger

from modelconverter.utils import is_hubai_available, resolve_path


class BenchmarkResult(NamedTuple):
    """Benchmark result, tuple (FPS, latency in ms, system and processor
    power in W, dsp utilization)"""

    fps: float
    latency: float | Literal["N/A"]
    power: tuple[float | None, float | None] = (None, None)
    dsp: float | None = None


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

        # HEADER
        header = [*self._base_header(results), *self._extra_header(results)]
        for field in header:
            table.add_column(f"[cyan]{field}")

        # ROWS
        for configuration, result in results:
            base_cells = list(self._base_row_cells(configuration, result))
            extra_cells = list(self._extra_row_cells(configuration, result))
            table.add_row(*base_cells, *extra_cells)

        Console().print(table)

    def _base_header(
        self,
        results: list[tuple[Configuration, BenchmarkResult]],
    ) -> list[str]:
        """Shared header cells."""
        return [*results[0][0].keys(), "fps", "latency (ms)"]

    def _base_row_cells(
        self,
        configuration: Configuration,
        result: BenchmarkResult,
    ) -> Iterable[str]:
        """Shared row cells for each result (configuration + fps +
        latency)."""
        # configuration values
        for x in configuration.values():
            yield f"[magenta]{x}"

        # fps
        fps_color = (
            "yellow"
            if 5 < result.fps < 15
            else "red"
            if result.fps < 5
            else "green"
        )
        yield f"[{fps_color}]{result.fps:.2f}"

        # latency
        if isinstance(result.latency, str):
            latency_color = "orange3"
            yield f"[{latency_color}]{result.latency}"
        else:
            latency_color = (
                "yellow"
                if 50 < result.latency < 100
                else "red"
                if result.latency > 100
                else "green"
            )
            yield f"[{latency_color}]{result.latency:.5f}"

    def _extra_header(
        self,
        results: list[tuple[Configuration, BenchmarkResult]],
    ) -> list[str]:
        """Columns to append after the base header (default: none)."""
        return []

    def _extra_row_cells(
        self,
        configuration: Configuration,
        result: BenchmarkResult,
    ) -> Iterable[str]:
        """Extra cells to append after the base row cells (default:

        none).
        """
        return []

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

        if not full:
            configurations = [{**self.default_configuration, **kwargs}]
        else:
            configurations = [
                {
                    **config,
                    **{k: v for k, v in kwargs.items() if k not in config},
                }
                for config in self.all_configurations  # add only kwarg keys that are not already there to not overwrite
            ]

        results: list[tuple[Configuration, BenchmarkResult]] = [
            (configuration, self.benchmark(configuration))
            for configuration in configurations
        ]

        # Clean up configuration keys: keep either benchmark_time or repetitions
        for configuration, _ in results:
            if configuration.get("benchmark_time"):
                items = list(configuration.items())
                configuration.clear()
                for k, v in items:
                    if k == "benchmark_time":
                        configuration["benchmark_time (s)"] = v
                    elif k != "repetitions":
                        configuration[k] = v
            else:
                configuration.pop("benchmark_time", None)

        self.print_results(results)
        if save:
            self.save_results(results)
