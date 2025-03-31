import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from modelconverter.utils import resolve_path, subprocess_run


class Analyzer(ABC):
    def __init__(self, dlc_model_path: str, image_dirs: Dict[str, str]):
        self.image_dirs: Dict[str, Path] = {}
        for key, value in image_dirs.items():
            self.image_dirs[key] = resolve_path(value, Path.cwd())

        self.dlc_model_path: Path = resolve_path(dlc_model_path, Path.cwd())
        self.model_name: str = self.dlc_model_path.stem
        self.input_sizes, self.data_types = self._get_input_sizes()
        self.output_sizes: Dict[str, List[int]] = self._get_output_sizes()

    @abstractmethod
    def analyze_layer_outputs(self, onnx_model_path: Path) -> None:
        pass

    @abstractmethod
    def analyze_layer_cycles(self) -> None:
        pass

    def _get_input_sizes(self) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        csv_path = Path("info.csv")
        subprocess_run(
            [
                "snpe-dlc-info",
                "-i",
                self.dlc_model_path,
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

        # Extract and load the relevant CSV part into a pandas DataFrame.
        relevant_csv_part = content[start_index:end_index].strip()
        df = pd.read_csv(io.StringIO(relevant_csv_part))
        sizes = {
            str(row["Input Name"]): list(
                map(int, str(row["Dimensions"]).split(","))
            )
            for _, row in df.iterrows()
        }
        data_types = {
            str(row["Input Name"]): str(row["Type"])
            for _, row in df.iterrows()
        }

        self._validate_inputs(sizes)

        return sizes, data_types

    def _validate_inputs(self, input_sizes: Dict[str, List[int]]) -> None:
        if len(self.image_dirs.keys()) == 1 and len(input_sizes.keys()) == 1:
            new_input_name = list(input_sizes.keys())[0]
            self.image_dirs = {new_input_name: self.image_dirs.popitem()[1]}
            return

        for name in self.image_dirs.keys():
            if name not in input_sizes.keys():
                raise ValueError(
                    f"The provided input name '{name}' does not match any of the DLC model's input names: {input_sizes.keys()}"
                )

    def _replace_bad_layer_names(self, layer_names: List[str]) -> List[str]:
        new_layer_names = []
        for layer_name in layer_names:
            ln = layer_name.replace("..", ".")
            ln = ln.replace("__", "_")
            ln = ln.replace(".", "_")
            ln = ln.replace("/", "_")
            ln = ln.strip("_")
            ln = ln.replace("_output_0", "")
            ln = ln.replace("(cycles)", "")

            new_layer_names.append(ln)

        return new_layer_names

    def _get_output_sizes(self) -> Dict[str, List[int]]:
        csv_path = Path("info.csv")
        subprocess_run(
            ["snpe-dlc-info", "-i", self.dlc_model_path, "-m", "-s", csv_path],
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

        df = pd.read_csv(
            csv_path, skiprows=start_row, nrows=end_row - start_row
        )
        csv_path.unlink()
        df = df.dropna()

        df = df[["Outputs", "Out Dims"]]
        df.columns = ["Name", "Shape"]
        df["Name"] = df["Name"].str.split(" ").str[0]
        df["Shape"] = df["Shape"].str.replace("x", ",")
        df["Shape"] = df["Shape"].apply(
            lambda x: [int(d) for d in x.split(",")]
        )
        df["Name"] = self._replace_bad_layer_names(list(df["Name"]))
        df.set_index("Name", inplace=True)
        output_sizes = df["Shape"].to_dict()

        return output_sizes
