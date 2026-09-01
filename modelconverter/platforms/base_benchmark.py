"""Platform-agnostic scaffolding for on-device benchmarking.

Every platform measures throughput and latency with its own
runtime, but the surrounding work is the same: resolve the model to
either a local file or a HubAI slug, run it once per configuration, and
report the collected numbers. That shared part lives in `Benchmark`,
which the per-platform benchmarks subclass.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TypeAlias, TypeVar

import polars as pl
from loguru import logger
from luxonis_ml.typing import PathType

from modelconverter.utils import is_hubai_model_variant_available, resolve_path

ConfigValue: TypeAlias = str | int | bool | None
Configuration: TypeAlias = dict[str, ConfigValue]

Result: TypeAlias = dict[str, float | str | None]

OptionT = TypeVar("OptionT", bound=str | int | bool)


def get_option(
    configuration: Configuration, key: str, option_type: type[OptionT]
) -> OptionT:
    """Read one benchmark option and check its type.

    Args:
        configuration: The options of a single benchmark run.
        key: The name of the option.
        option_type: The necessary type of the option.

    Returns:
        The value of the option.

    Raises:
        TypeError: If the option is missing or has a different type.

    """
    value = configuration.get(key)
    if not isinstance(value, option_type):
        raise TypeError(
            f"The benchmark option '{key}' must be of type "
            f"'{option_type.__name__}', got {value!r}."
        )
    return value


def get_optional_option(
    configuration: Configuration, key: str, option_type: type[OptionT]
) -> OptionT | None:
    """Read one benchmark option that may also be unset.

    Args:
        configuration: The options of a single benchmark run.
        key: The name of the option.
        option_type: The necessary type of the option.

    Returns:
        The value of the option, or ``None`` if it is not set.

    Raises:
        TypeError: If the option has a different type.

    """
    if configuration.get(key) is None:
        return None
    return get_option(configuration, key, option_type)


class Benchmark(ABC):
    """Base class for benchmarking a converted model on a device.

    Subclasses implement the platform-specific measurement in
    `benchmark` and describe what to measure it with in
    `default_configuration` and `all_configurations`.
    """

    _VALID_EXTENSIONS = (".tar.xz", ".blob", ".dlc")
    _HUB_MODEL_PATTERN = re.compile(
        r"^(?:([^/]+)/)?([^:]+):([^:]+)(?::(.+))?$"
    )

    def __init__(self, model_path: str):
        """Resolve the model to benchmark.

        Args:
            model_path: Either a path to a local model file ending in
                one of the supported extensions, or a HubAI model slug
                of the form
                ``[team_name/]model_name:variant[:model_instance]``.

        Raises:
            ValueError: If ``model_path`` is neither a supported model
                file nor a slug of a model available on HubAI.

        """
        self.model_path: PathType
        self._model_instance: str | None

        if any(model_path.endswith(ext) for ext in self._VALID_EXTENSIONS):
            self.model_path = resolve_path(model_path, Path.cwd())
            self.model_name = self.model_path.stem
            self._model_instance = None
        else:
            hub_match = self._HUB_MODEL_PATTERN.match(model_path)
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
            hub_model_identifier = (
                f"{team_name}/{model_name}" if team_name else model_name
            )
            if is_hubai_model_variant_available(
                hub_model_identifier, model_variant
            ):
                self.model_path = model_path
                self.model_name = model_name
                self._model_variant = model_variant
                self._model_instance = model_instance
                self._hub_model_identifier = hub_model_identifier
            else:
                raise ValueError(
                    f"Model {team_name + '/' if team_name else ''}{model_name}:{model_variant}{':' + model_instance if model_instance else ''} not found in HubAI."
                )

    @abstractmethod
    def benchmark(self, configuration: Configuration) -> Result:
        """Run a single benchmark with the given configuration.

        Args:
            configuration: The options to benchmark with.

        Returns:
            The measured metrics. Must contain a ``"fps"`` entry and may
            contain a ``"latency"`` entry.

        """

    @property
    @abstractmethod
    def default_configuration(self) -> Configuration:
        """Return the configuration a plain benchmark run uses."""

    @property
    @abstractmethod
    def all_configurations(self) -> list[Configuration]:
        """Return the configurations a full benchmark run sweeps."""

    def print_results(
        self, results: list[tuple[Configuration, Result]]
    ) -> None:
        """Print the benchmark results as a table.

        The table is transposed relative to the results: one column per
        run, one row per configuration option and measured metric.

        Args:
            results: The configuration/result pairs to print. Must not
                be empty.

        """
        assert results, "No results to print"

        from rich import box
        from rich.console import Console
        from rich.table import Table

        table = Table(
            title=f"Benchmark Results for [yellow]{self.model_name}",
            box=box.ROUNDED,
        )

        header = [*self._base_header(results), *self._extra_header(results)]
        result_rows = [
            [
                *self._base_row_cells(configuration, result),
                *self._extra_row_cells(configuration, result),
            ]
            for configuration, result in results
        ]

        table.add_column("[cyan]metric")
        for index in range(1, len(result_rows) + 1):
            table.add_column(f"[cyan]run {index}")

        for field, *cells in zip(header, *result_rows, strict=True):
            table.add_row(f"[cyan]{field}", *cells)

        Console().print(table)

    def _base_header(
        self,
        results: list[tuple[Configuration, Result]],
    ) -> list[str]:
        """Shared header cells."""
        return [*results[0][0].keys(), "fps", "latency (ms)"]

    def _base_row_cells(
        self,
        configuration: Configuration,
        result: Result,
    ) -> Iterable[str]:
        """Shared row cells for each result (configuration + fps +
        latency).
        """
        # configuration values
        for x in configuration.values():
            yield f"[magenta]{x}"

        fps = result.get("fps")
        if not isinstance(fps, int | float):
            raise TypeError(f"The benchmark measured no FPS, got {fps!r}.")
        fps_color = "yellow" if 5 < fps < 15 else "red" if fps < 5 else "green"
        yield f"[{fps_color}]{fps:.2f}"

        latency = result.get("latency")
        if isinstance(latency, int | float):
            latency_color = (
                "yellow"
                if 50 < latency < 100
                else "red"
                if latency > 100
                else "green"
            )
            yield f"[{latency_color}]{latency:.2f}"
        else:
            yield f"[orange3]{latency or 'N/A'}"

    def _extra_header(
        self,
        results: list[tuple[Configuration, Result]],
    ) -> list[str]:
        """Return columns to append after the base header.

        Args:
            results: All benchmark results, as configuration/result
                pairs.

        Returns:
            The extra columns. Empty by default.

        """
        return []

    def _extra_row_cells(
        self,
        configuration: Configuration,
        result: Result,
    ) -> Iterable[str]:
        """Return extra cells to append after the base row cells.

        Args:
            configuration: The configuration the result was produced
                with.
            result: A single benchmark result.

        Returns:
            The extra cells. Empty by default.

        """
        return []

    def save_results(
        self, results: list[tuple[Configuration, Result]]
    ) -> None:
        """Save the benchmark results to a CSV file.

        The file is named ``<model_name>_benchmark_results.csv`` and is
        written to the current working directory. A nested ``power``
        column is first split into ``power_system`` and
        ``power_processor``.

        Args:
            results: The configuration/result pairs to save. Must not
                be empty.

        """
        assert results, "No results to save"
        df = pl.DataFrame(
            [configuration | result for configuration, result in results]
        )

        # Split nested power list into two CSV-friendly columns
        if "power" in df.columns and isinstance(df.schema["power"], pl.List):
            df = df.with_columns(
                power_system=pl.col("power").list.get(0),
                power_processor=pl.col("power").list.get(1),
            ).drop("power")

        file = f"{self.model_name}_benchmark_results.csv"
        df.write_csv(file)
        logger.info(f"Benchmark results saved to {file}.")

    def run(
        self, full: bool = True, save: bool = False, **kwargs: ConfigValue
    ) -> None:
        """Benchmark the model and report the results.

        Args:
            full: If ``True``, benchmark every configuration in
                `all_configurations`. If ``False``, benchmark only
                `default_configuration`.
            save: If ``True``, the results are written to a CSV file in
                addition to being printed.
            **kwargs: Overrides for individual configuration options.
                An override of an option with a non-``None`` default is
                cast to the type of that default, unless the default is
                a boolean. In a full run, an override is only applied to
                configurations that do not set the option themselves.

        """
        logger.info(f"Running benchmarking for {self.model_name}")
        # `all_configurations` names only the options it varies, so the
        # rest have to come from the defaults. An explicit null stays
        # null.
        for key, default in self.default_configuration.items():
            value = kwargs.get(key, default)
            # `bool` accepts any object, so it turns the string "false"
            # into `True`. A boolean option keeps its value and
            # `get_option` refuses a value of the wrong type.
            if (
                default is not None
                and value is not None
                and not isinstance(default, bool)
            ):
                value = type(default)(value)
            kwargs[key] = value

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

        results: list[tuple[Configuration, Result]] = []
        for configuration in configurations:
            logger.info(f"Running with configuration: {configuration}")
            results.append((configuration, self.benchmark(configuration)))

        # Clean up configuration keys: keep either benchmark_time or repetitions
        for configuration, _ in results:
            benchmark_time = configuration.get("benchmark_time")
            if isinstance(benchmark_time, int) and benchmark_time > 0:
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
