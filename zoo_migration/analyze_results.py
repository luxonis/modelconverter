from collections.abc import Iterator
from enum import Enum
from functools import reduce
from pathlib import Path

import polars as pl
from cyclopts import App
from rich import print
from rich.console import RenderableType, group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

app = App("analyze_results", "Analyze results from a CSV file")


def _filter_error(column: str, value: str) -> pl.Expr:
    return pl.col(column).str.contains(value)


class Reason(Enum):
    modelconverter = "modelconverter"
    degradation = "Degradation"
    nan = "NaN"
    sizes = "sizes"
    multi_stage = "Multi-stage"
    item = "item()"
    no_images = "No images"
    only_single_input = "Only single input"
    parent_not_found = "Parent not found"


def print_info(df: pl.DataFrame, reason: Reason) -> None:
    for row in df.iter_rows(named=True):

        @group()
        def render(row: dict[str, str]) -> Iterator[RenderableType]:
            yield f"Model ID: {row['model_id']}"
            yield f"Variant ID: {row['variant_id']}"
            yield f"Instance ID: {row['instance_id']}"
            yield Rule()
            yield Text(row["error"], style="bold red")

        print(Panel(render(row), title=f"Failed Conversion ({reason.value})"))


@app.default
def main(
    csv_path: Path, reason: Reason | None = None, info: bool = False
) -> None:
    df = pl.read_csv(csv_path)
    failed = df.filter(pl.col("status") == "failed")
    print(f"Successful conversions: {len(df) - len(failed)}/{len(df)}")
    print(f"Failed conversions: {len(failed)}/{len(df)}")
    subfails_len = 0
    instance_ids = set()
    if reason is not None:
        errors = [reason.value]
    else:
        errors = [
            "modelconverter",
            "Degradation",
            "NaN",
            "sizes",
            "Multi-stage",
            "item()",
            "No images",
            "Only single input",
            "Parent not found",
        ]
    for error in errors:
        if error == "modelconverter":
            _df = failed.filter(
                _filter_error("error", "modelconverter")
                & ~_filter_error("error", "No images")
            )
        elif error == "Degradation":
            _df = failed.filter(
                _filter_error("error", "Degradation")
                & ~_filter_error("error", "NaN")
            )
        else:
            _df = failed.filter(_filter_error("error", error))
        instances = set(_df.select(pl.col("instance_id")).unique().to_series())
        if instances & instance_ids:
            print(error)
            raise ValueError(
                f"Instance IDs {instances & instance_ids} already seen in previous errors."
            )
        print(f"Failed conversions ({error}): {len(_df)}/{len(failed)}")
        if info:
            print_info(_df, Reason(error))
        subfails_len += len(_df)
        instance_ids |= instances

    if subfails_len != len(failed):
        _df = failed.filter(
            reduce(
                lambda x, y: x & y,
                (~_filter_error("error", error) for error in errors),
            )
        )
        print(f"Failed conversions (other): {len(_df)}/{len(failed)}")


if __name__ == "__main__":
    app()
