from collections.abc import Iterable, Iterator
from pathlib import Path

import ml_dtypes
import numpy as np
import numpy.typing as npt
import onnx
from onnx.external_data_helper import ExternalDataInfo, uses_external_data


def ensure_onnx_helper_compatibility() -> None:
    helper = onnx.helper

    def _convert_scalar(
        value: float, dtype: npt.DTypeLike, container: npt.DTypeLike
    ) -> int:
        arr = np.asarray(value, dtype=dtype)
        return arr.view(container).item()

    if not hasattr(helper, "float32_to_bfloat16"):
        helper.float32_to_bfloat16 = lambda value: _convert_scalar(  # type: ignore[attr-defined]
            value, ml_dtypes.bfloat16, np.uint16
        )

    if not hasattr(helper, "float32_to_float8e4m3"):
        dtype_map = {
            (False, False): ml_dtypes.float8_e4m3,
            (True, False): ml_dtypes.float8_e4m3fn,
            (True, True): ml_dtypes.float8_e4m3fnuz,
            (False, True): ml_dtypes.float8_e4m3b11fnuz,
        }

        def float32_to_float8e4m3(
            value: float, *, fn: bool = True, uz: bool = False
        ) -> int:
            return _convert_scalar(value, dtype_map[(fn, uz)], np.uint8)

        helper.float32_to_float8e4m3 = float32_to_float8e4m3  # type: ignore[attr-defined]


def save_onnx_model(
    model: onnx.ModelProto,
    output_path: str | Path,
    *,
    save_as_external_data: bool = False,
    location: str | None = None,
) -> None:
    output_path = Path(output_path)

    if save_as_external_data:
        external_data_path = output_path.with_name(
            location or f"{output_path.name}_data"
        )
        if external_data_path.exists():
            external_data_path.unlink()
        onnx.save(
            model,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
            size_threshold=0,
            convert_attribute=False,
        )
        return

    onnx.save(model, str(output_path))


def _iter_node_tensors(
    nodes: Iterable[onnx.NodeProto],
) -> Iterator[onnx.TensorProto]:
    """Yields the tensors reachable from ``nodes``.

    Besides the subgraphs nested in a node (the bodies of ``If``,
    ``Loop``, ``Scan``, ...), an attribute can hold a tensor directly --
    the value of a ``Constant``, say -- and ONNX writes those out
    externally too when the model is converted with
    ``convert_attribute=True``.
    """
    for node in nodes:
        for attribute in node.attribute:
            yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("g"):
                yield from _iter_tensors(attribute.g)
            for subgraph in attribute.graphs:
                yield from _iter_tensors(subgraph)


def _iter_tensors(graph: onnx.GraphProto) -> Iterator[onnx.TensorProto]:
    """Yields every initializer of ``graph``, including those held by its
    nodes."""
    yield from graph.initializer
    for sparse_tensor in graph.sparse_initializer:
        yield sparse_tensor.values
        yield sparse_tensor.indices

    yield from _iter_node_tensors(graph.node)


def _iter_model_tensors(model: onnx.ModelProto) -> Iterator[onnx.TensorProto]:
    """Yields every tensor of ``model`` that may carry external data.

    Local functions are walked as well: their nodes are not part of the
    main graph, but ONNX's own loader resolves their external data.
    """
    yield from _iter_tensors(model.graph)
    for function in model.functions:
        yield from _iter_node_tensors(function.node)


def get_external_data_paths(model_path: str | Path) -> list[Path]:
    """Returns every companion file holding the model's external tensor
    data.

    A model saved with ``all_tensors_to_one_file=False`` keeps one file
    per tensor, so there is not necessarily just one.
    """
    model_path = Path(model_path)
    model = onnx.load(str(model_path), load_external_data=False)
    model_dir = model_path.parent.resolve()

    paths: list[Path] = []
    for tensor in _iter_model_tensors(model):
        if not uses_external_data(tensor):
            continue
        location = ExternalDataInfo(tensor).location
        if not location:
            continue
        candidate = (model_path.parent / location).resolve()
        try:
            candidate.relative_to(model_dir)
        except ValueError as exc:
            raise ValueError(
                "ONNX external data location must stay within the "
                f"model directory: {location}"
            ) from exc
        if candidate not in paths:
            paths.append(candidate)
    return paths


def has_external_data(model_path: str | Path) -> bool:
    """Whether the model keeps any of its tensors in companion files.

    Callers that re-save the model consolidate every tensor into one new
    file, so they only need to know *whether* there is external data --
    not where it currently lives.
    """
    return bool(get_external_data_paths(model_path))
