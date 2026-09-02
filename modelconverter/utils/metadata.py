"""Reading of input and output metadata from model files.

Every conversion platform speaks a different model format, so the rest
of modelconverter asks for shapes and data types through the single
`get_metadata` entry point, which dispatches on the file suffix: ONNX,
OpenVINO IR (RVC2/RVC3), SNPE DLC (RVC4), Hailo HAR and TFLite. The
readers for the platform-specific formats rely on tooling that is only
present inside that platform's container, so they are imported lazily.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import onnx

from modelconverter.utils.subprocess import subprocess_run
from modelconverter.utils.types import DataType

if TYPE_CHECKING:
    import tflite


@dataclass
class Metadata:
    """Shapes and data types of a model's inputs and outputs.

    Attributes:
        input_shapes: Shape of each input, keyed by input name.
        input_dtypes: Data type of each input, keyed by input name.
        output_shapes: Shape of each output, keyed by output name.
        output_dtypes: Data type of each output, keyed by output name.

    """

    input_shapes: dict[str, list[int]]
    input_dtypes: dict[str, DataType]
    output_shapes: dict[str, list[int]]
    output_dtypes: dict[str, DataType]


def get_metadata(model_path: Path) -> Metadata:
    """Read the metadata of a model, whatever format it is in.

    The format is taken from the file suffix: ``.onnx``, ``.xml`` or
    ``.bin`` (OpenVINO IR), ``.dlc`` or ``.csv`` (SNPE), ``.hef`` or
    ``.har`` (Hailo), and ``.tflite``.

    Args:
        model_path: Path to the model file. For an IR model either the
            ``.xml`` or the ``.bin`` file may be given, the other one
            being derived from it.

    Returns:
        Metadata of the model's inputs and outputs.

    Raises:
        ValueError: If the suffix is not one of the supported formats,
            or if an ONNX or IR model cannot be read. The other
            formats let their reader's own errors propagate, such as
            ``FileNotFoundError`` for a missing file.

    """
    suffix = model_path.suffix
    if suffix in {".dlc", ".csv"}:
        return _get_metadata_dlc(model_path)
    if suffix == ".onnx":
        return _get_metadata_onnx(model_path)
    if suffix in {".xml", ".bin"}:
        if suffix == ".xml":
            xml_path = model_path
            bin_path = model_path.with_suffix(".bin")
        else:
            bin_path = model_path
            xml_path = model_path.with_suffix(".xml")
        return _get_metadata_ir(bin_path, xml_path)
    if suffix in {".hef", ".har"}:
        return _get_metadata_hailo(model_path)
    if suffix == ".tflite":
        return _get_metadata_tflite(model_path)
    raise ValueError(f"Unsupported model format: {suffix}")


def _get_metadata_dlc(path: Path) -> Metadata:
    import polars as pl

    if path.suffix == ".csv":
        csv_path = path
    else:
        csv_path = path.with_suffix(".info.csv")
        if (
            not csv_path.exists()
            or csv_path.stat().st_mtime < path.stat().st_mtime
        ):
            subprocess_run(
                ["snpe-dlc-info", "-i", path, "-s", csv_path], silent=True
            )
    content = csv_path.read_text()

    metadata = {}

    for typ in ["input", "output"]:
        header_pattern = f"{typ.capitalize()} Name"

        start_index = content.find(header_pattern)
        if start_index == -1:
            continue

        line_start = content.rfind("\n", 0, start_index) + 1
        possible_endings = []

        if typ == "input":
            output_idx = content.find(
                "Output Name", start_index + len(header_pattern)
            )
            if output_idx != -1:
                possible_endings.append(output_idx)
        else:
            unconsumed_idx = content.find(
                "Unconsumed Tensor Name", start_index + len(header_pattern)
            )
            total_idx = content.find(
                "Total parameters:", start_index + len(header_pattern)
            )

            if unconsumed_idx != -1:
                possible_endings.append(unconsumed_idx)
            if total_idx != -1:
                possible_endings.append(total_idx)

        if possible_endings:
            end_index = min(possible_endings)
        else:
            end_index = len(content)

        section = content[line_start:end_index].strip()
        if not section:  # pragma: no cover
            continue

        lines = section.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped and not all(c in "-|+= " for c in stripped):
                cleaned_line = line.strip()
                if cleaned_line.startswith("|") and cleaned_line.endswith("|"):
                    cleaned_line = cleaned_line[1:-1].strip()
                    import re

                    cleaned_line = re.sub(r"\s*\|\s*", ",", cleaned_line)
                cleaned_lines.append(cleaned_line)

        relevant_csv_part = "\n".join(cleaned_lines)

        if not relevant_csv_part.strip():  # pragma: no cover
            continue

        df = pl.read_csv(io.StringIO(relevant_csv_part))

        shapes = df.select(
            [
                pl.col(f"{typ.capitalize()} Name"),
                pl.col("Dimensions").str.split(",").cast(pl.List(pl.Int64)),
            ]
        ).to_dict(as_series=False)
        metadata[f"{typ}_shapes"] = dict(
            zip(
                map(str, shapes[f"{typ.capitalize()} Name"]),
                shapes["Dimensions"],
                strict=True,
            )
        )

        dtypes = df.select(
            [pl.col(f"{typ.capitalize()} Name"), pl.col("Type")]
        ).to_dict(as_series=False)
        metadata[f"{typ}_dtypes"] = {
            str(name): DataType.from_dlc_dtype(dtype)
            for name, dtype in zip(
                dtypes[f"{typ.capitalize()} Name"], dtypes["Type"], strict=True
            )
        }

    return Metadata(**metadata)


def _get_metadata_ir(bin_path: Path, xml_path: Path) -> Metadata:
    from openvino.runtime import Core

    ie = Core()
    try:
        model = ie.read_model(model=str(xml_path), weights=str(bin_path))
    except Exception as e:
        raise ValueError(
            f"Failed to load IR model: `{bin_path}` and `{xml_path}`"
        ) from e

    input_shapes = {}
    input_dtypes = {}
    output_shapes = {}
    output_dtypes = {}

    for inp in model.inputs:
        name = next(iter(inp.names))
        input_shapes[name] = list(inp.shape)
        input_dtypes[name] = DataType.from_ir_runtime_dtype(
            inp.element_type.get_type_name()
        )
    for output in model.outputs:
        name = next(iter(output.names))
        output_shapes[name] = list(output.shape)
        output_dtypes[name] = DataType.from_ir_runtime_dtype(
            output.element_type.get_type_name()
        )

    return Metadata(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def _get_metadata_onnx(onnx_path: Path) -> Metadata:
    try:
        model = onnx.load(str(onnx_path))
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model: `{onnx_path}`") from e

    input_shapes = {}
    input_dtypes = {}
    output_shapes = {}
    output_dtypes = {}

    for inp in model.graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        input_shapes[inp.name] = shape
        input_dtypes[inp.name] = DataType.from_onnx_dtype(
            inp.type.tensor_type.elem_type
        )

    for output in model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        output_shapes[output.name] = shape
        output_dtypes[output.name] = DataType.from_onnx_dtype(
            output.type.tensor_type.elem_type
        )

    return Metadata(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def _get_metadata_tflite(model_path: Path) -> Metadata:
    import tflite

    with open(model_path, "rb") as f:
        data = f.read()

    subgraph = tflite.Model.GetRootAsModel(data, 0).Subgraphs(0)

    if subgraph is None:
        raise ValueError("Failed to load TFLite model.")

    input_shapes = {}
    input_dtypes = {}
    output_shapes = {}
    output_dtypes = {}

    for i in range(subgraph.InputsLength()):
        name, shape, dtype = _read_tflite_tensor(subgraph, subgraph.Inputs(i))
        input_shapes[name] = shape
        input_dtypes[name] = dtype

    for i in range(subgraph.OutputsLength()):
        name, shape, dtype = _read_tflite_tensor(subgraph, subgraph.Outputs(i))
        output_shapes[name] = shape
        output_dtypes[name] = dtype

    return Metadata(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def _get_metadata_hailo(model_path: Path) -> Metadata:
    from hailo_sdk_client import ClientRunner

    input_shapes = {}
    input_dtypes = {}
    output_shapes = {}
    output_dtypes = {}
    runner = ClientRunner(hw_arch="hailo8", har=str(model_path))
    for params in runner.get_hn_dict()["layers"].values():
        if params["type"] in ["input_layer", "output_layer"]:
            name = params["original_names"][0]
            shape = list(params["input_shapes"][0])
            if shape[0] == -1:
                shape[0] = 1
            # TODO: should find a way to read the dtypes from Hailo SDK instead of hard-coding int8
            if params["type"] == "input_layer":
                input_shapes[name] = shape
                input_dtypes[name] = DataType.INT8
            else:
                output_shapes[name] = shape
                output_dtypes[name] = DataType.INT8

    return Metadata(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def _read_tflite_tensor(
    subgraph: "tflite.SubGraph", tensor_index: int
) -> tuple[str, list[int], DataType]:
    tensor = subgraph.Tensors(tensor_index)
    if tensor is None:
        raise ValueError(
            f"TFLite model has no tensor at index {tensor_index}."
        )
    name = tensor.Name()
    if name is None:
        raise ValueError(f"TFLite tensor at index {tensor_index} has no name.")
    return (
        name.decode("utf-8"),
        [tensor.Shape(j) for j in range(tensor.ShapeLength())],
        DataType.from_tensorflow_dtype(tensor.Type()),
    )
