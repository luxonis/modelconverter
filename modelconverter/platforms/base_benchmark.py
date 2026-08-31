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
    """Reads one benchmark option and checks its type.

    @type configuration: Configuration
    @param configuration: The options of a single benchmark run.
    @type key: str
    @param key: The name of the option.
    @type option_type: type[OptionT]
    @param option_type: The necessary type of the option.
    @rtype: OptionT
    @return: The value of the option.
    @raise TypeError: If the option is missing or has a different type.
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
    """As L{get_option}, but the option can also be unset.

    @type configuration: Configuration
    @param configuration: The options of a single benchmark run.
    @type key: str
    @param key: The name of the option.
    @type option_type: type[OptionT]
    @param option_type: The necessary type of the option.
    @rtype: OptionT | None
    @return: The value of the option, or C{None} if it is not set.
    @raise TypeError: If the option has a different type.
    """
    if configuration.get(key) is None:
        return None
    return get_option(configuration, key, option_type)


class Benchmark(ABC):
    _VALID_EXTENSIONS = (".tar.xz", ".blob", ".dlc")
    _HUB_MODEL_PATTERN = re.compile(
        r"^(?:([^/]+)/)?([^:]+):([^:]+)(?::(.+))?$"
    )

    def __init__(self, model_path: str):
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
        self, results: list[tuple[Configuration, Result]]
    ) -> None:
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
        latency)."""
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
        """Columns to append after the base header (default: none)."""
        return []

    def _extra_row_cells(
        self,
        configuration: Configuration,
        result: Result,
    ) -> Iterable[str]:
        """Extra cells to append after the base row cells (default:

        none).
        """
        return []

    def save_results(
        self, results: list[tuple[Configuration, Result]]
    ) -> None:
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
