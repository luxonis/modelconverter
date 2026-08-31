"""Base class of the per-platform model analyzers.

An analyzer inspects a converted model layer by layer on a connected
device: it compares the output of every layer against the ONNX model the
conversion started from, and measures the cycles each layer takes. Only
the RVC4 package implements it, on top of the SNPE tooling shipped in
the RVC4 container.
"""

import io
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

from modelconverter.utils import resolve_path, subprocess_run


class Analyzer(ABC):
    """Base class for the layer-wise analysis of a DLC model.

    Reads the names and shapes of the model's inputs and outputs, and
    the data types of its inputs, from ``snpe-dlc-info``, and checks
    the directories holding the images the analysis is run on.

    Attributes:
        image_dirs: Resolved image directory per model input.
        dlc_model_path: Resolved path to the analyzed DLC model.
        model_name: Name of the model, taken from its file name.
        input_sizes: Shape of each input of the DLC model.
        data_types: Data type of each input of the DLC model.
        output_sizes: Shape of each output layer of the DLC model.

    """

    def __init__(self, dlc_model_path: str, image_dirs: dict[str, str]):
        """Initialize the analyzer and read the model's layer shapes.

        Args:
            dlc_model_path: Path to the DLC model to analyze.
            image_dirs: Directory of input images per model input, keyed
                by input name. All of them must hold the same number of
                files. A single directory keyed by any name is matched
                to the only input of a single-input model.

        Raises:
            ValueError: If the directories do not all hold the same
                number of files, if there are no directories, or if an
                input name matches none of the model's inputs.

        """
        self._image_dirs: dict[str, Path] = {}
        for key, value in image_dirs.items():
            self._image_dirs[key] = resolve_path(value, Path.cwd())

        self._check_dir_sizes()
        self._dlc_model_path: Path = resolve_path(dlc_model_path, Path.cwd())
        self._model_name: str = self._dlc_model_path.stem
        self._input_sizes, self._data_types = self._get_input_sizes()
        self._output_sizes: dict[str, list[int]] = self._get_output_sizes()

    @abstractmethod
    def analyze_layer_outputs(self, onnx_model_path: Path) -> None:
        """Compare the layer outputs against the original ONNX model.

        Args:
            onnx_model_path: Path to the ONNX model the analyzed model
                was converted from.

        """

    @abstractmethod
    def analyze_layer_cycles(self) -> None:
        """Measure the cycles each layer of the model takes."""

    def _get_input_sizes(self) -> tuple[dict[str, list[int]], dict[str, str]]:
        csv_path = Path("info.csv")
        subprocess_run(
            [
                "snpe-dlc-info",
                "-i",
                self._dlc_model_path,
                "-s",
                csv_path,
            ],
            silent=True,
        )
        content = csv_path.read_text()
        csv_path.unlink()

        start_marker = "Input Name,Dimensions,Type,Encoding Info"
        end_marker = "Output Name,Dimensions,Type,Encoding Info"
        start_index = content.find(start_marker)
        end_index = content.find(end_marker, start_index)

        relevant_csv_part = content[start_index:end_index].strip()
        df = pl.read_csv(io.StringIO(relevant_csv_part))
        sizes = {
            str(row["Input Name"]): list(
                map(int, str(row["Dimensions"]).split(","))
            )
            for row in df.to_dicts()
        }
        data_types = {
            str(row["Input Name"]): str(row["Type"]) for row in df.to_dicts()
        }

        self._validate_inputs(sizes)

        return sizes, data_types

    def _validate_inputs(self, input_sizes: dict[str, list[int]]) -> None:
        if len(self._image_dirs.keys()) == 1 and len(input_sizes.keys()) == 1:
            new_input_name = next(iter(input_sizes.keys()))
            self._image_dirs = {new_input_name: self._image_dirs.popitem()[1]}
            return

        for name in self._image_dirs:
            if name not in input_sizes:
                raise ValueError(
                    f"The provided input name '{name}' does not match any of the DLC model's input names: {input_sizes.keys()}"
                )

    def _replace_bad_layer_names(self, layer_names: list[str]) -> list[str]:
        new_layer_names = []
        for layer_name in layer_names:
            ln = self._replace_bad_layer_name(layer_name)
            new_layer_names.append(ln)

        return new_layer_names

    def _replace_bad_layer_name(self, ln: str) -> str:
        ln = ln.replace("..", ".")
        ln = ln.replace("__", "_")
        ln = ln.replace(".", "_")
        ln = ln.replace("/", "_")
        ln = ln.replace("_output_0", "")
        ln = ln.replace("(cycles)", "")

        return ln.strip("_")

    def _get_output_sizes(self) -> dict[str, list[int]]:
        csv_path = Path("info.csv")
        subprocess_run(
            [
                "snpe-dlc-info",
                "-i",
                self._dlc_model_path,
                "-m",
                "-s",
                csv_path,
            ],
            silent=True,
        )
        lines = csv_path.read_text().splitlines()
        start_row = next(
            i
            for i, line in enumerate(lines)
            if line.strip().startswith("Id,Name,Type,Inputs,Outputs")
        )
        end_row = next(
            i
            for i, line in enumerate(lines[start_row:], start=start_row)
            if line.strip().startswith("Note:")
        )
        csv_portion = "\n".join(lines[start_row:end_row])
        df = pl.read_csv(io.StringIO(csv_portion))
        df = df.drop_nulls()

        df = df.select(["Outputs", "Out Dims"])
        df = df.rename({"Outputs": "Name", "Out Dims": "Shape"})

        df = df.with_columns(
            pl.col("Name").str.split(" ").list.first().alias("Name"),
            pl.col("Shape")
            .str.split("x")
            .list.eval(pl.element().cast(int))
            .alias("Shape"),
        )

        names = self._replace_bad_layer_names(df["Name"].to_list())
        df = df.with_columns(pl.Series("Name", names))
        csv_path.unlink()

        return {row["Name"]: row["Shape"] for row in df.to_dicts()}

    def _check_dir_sizes(self) -> None:
        dir_lengths = [
            len(list(v.iterdir())) for v in self._image_dirs.values()
        ]

        if len(set(dir_lengths)) > 1:
            raise ValueError(
                "All directories must have the same number of files."
            )
        if len(dir_lengths) == 0:
            raise ValueError("All directories must have at least one file.")
