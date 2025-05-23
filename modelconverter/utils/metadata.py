import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import onnx

from modelconverter.utils.subprocess import subprocess_run
from modelconverter.utils.types import DataType


@dataclass
class Metadata:
    input_shapes: dict[str, list[int]]
    input_dtypes: dict[str, DataType]
    output_shapes: dict[str, list[int]]
    output_dtypes: dict[str, DataType]


def get_metadata(model_path: Path) -> Metadata:
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
        csv_path = Path("info.csv")
        subprocess_run(
            ["snpe-dlc-info", "-i", path, "-s", csv_path], silent=True
        )
    content = csv_path.read_text()

    metadata = {}

    for typ in ["input", "output"]:
        start_marker = f"{typ.capitalize()} Name,Dimensions,Type,Encoding Info"
        if typ == "input":
            end_marker = "Output Name,Dimensions,Type,Encoding Info"
        else:
            end_marker = "Total parameters:"
        start_index = content.find(start_marker)
        end_index = content.find(end_marker, start_index)

        relevant_csv_part = content[start_index:end_index].strip()
        if not relevant_csv_part:
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
        tensor = subgraph.Tensors(subgraph.Inputs(i))
        input_shapes[tensor.Name().decode("utf-8")] = (  # type: ignore
            tensor.ShapeAsNumpy().tolist()  # type: ignore
        )
        input_dtypes[tensor.Name().decode("utf-8")] = (  # type: ignore
            DataType.from_tensorflow_dtype(tensor.Type())  # type: ignore
        )

    for i in range(subgraph.OutputsLength()):
        tensor = subgraph.Tensors(subgraph.Outputs(i))
        output_shapes[tensor.Name().decode("utf-8")] = (  # type: ignore
            tensor.ShapeAsNumpy().tolist()  # type: ignore
        )
        output_dtypes[tensor.Name().decode("utf-8")] = (  # type: ignore
            DataType.from_tensorflow_dtype(tensor.Type())  # type: ignore
        )

    return Metadata(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def _get_metadata_hailo(model_path: Path) -> Metadata:
    from modelconverter.packages.hailo.exporter import ClientRunner

    input_shapes = {}
    input_dtypes = {}
    output_shapes = {}
    output_dtypes = {}
    runner = ClientRunner(hw_arch="hailo8", har=str(model_path))
    for params in cast(Callable[..., dict], runner.get_hn_dict)()[
        "layers"
    ].values():
        if params["type"] in ["input_layer", "output_layer"]:
            name = params["original_names"][0]
            shape = list(params["input_shapes"][0])
            if shape[0] == -1:
                shape[0] = 1
            if params["type"] == "input_layer":
                input_shapes[name] = shape
                input_dtypes[name] = None
            else:
                output_shapes[name] = shape
                output_dtypes[name] = None

    return Metadata(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
