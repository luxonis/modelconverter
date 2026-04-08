from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelconverter.packages.base_exporter import Exporter
from modelconverter.utils.config import generate_renamed_onnx
from modelconverter.utils.onnx_compatibility import (
    ensure_onnx_helper_compatibility,
    save_onnx_model,
)
from modelconverter.utils.types import DataType


@pytest.mark.parametrize(
    ("tensor_name", "expected"),
    [
        ("BFLOAT16", DataType.BFLOAT16),
        ("INT4", DataType.INT4),
        ("UINT4", DataType.UINT4),
    ],
)
def test_extended_onnx_dtype_support(tensor_name: str, expected: DataType):
    if not hasattr(TensorProto, tensor_name):
        pytest.skip(f"{tensor_name} is not available in this ONNX version")
    assert (
        DataType.from_onnx_dtype(getattr(TensorProto, tensor_name)) == expected
    )


def test_simplify_onnx_falls_back_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import onnx
    import onnxsim

    input_tensor = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [1, 4]
    )
    output_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 4]
    )
    node = helper.make_node("Identity", inputs=["input0"], outputs=["output0"])
    model = helper.make_model(
        helper.make_graph(
            [node], "SimplifyFallbackModel", [input_tensor], [output_tensor]
        )
    )
    input_path = tmp_path / "fallback.onnx"
    onnx.save(model, input_path)

    monkeypatch.setattr(
        onnxsim,
        "simplify",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    class DummyExporter:
        input_model = input_path
        _attach_suffix = staticmethod(Exporter._attach_suffix)

    assert Exporter.simplify_onnx(DummyExporter()) == input_path


def test_generate_renamed_onnx_overwrites_external_data(tmp_path: Path):
    input_tensor = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [1, 1024]
    )
    output_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 1024]
    )
    bias_tensor = numpy_helper.from_array(
        np.arange(1024, dtype=np.float32).reshape(1, 1024), name="bias"
    )
    node = helper.make_node(
        "Add", inputs=["input0", "bias"], outputs=["output0"]
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "ExternalDataModel",
            [input_tensor],
            [output_tensor],
            initializer=[bias_tensor],
        ),
        producer_name="DummyModelProducer",
    )

    input_path = tmp_path / "external_input.onnx"
    output_path = tmp_path / "external_output.onnx"

    save_onnx_model(
        model,
        input_path,
        save_as_external_data=True,
        location=f"{input_path.name}_data",
    )
    assert input_path.with_name(f"{input_path.name}_data").exists()

    generate_renamed_onnx(input_path, {"output0": "renamed0"}, output_path)
    assert output_path.with_name(f"{output_path.name}_data").exists()

    generate_renamed_onnx(input_path, {"output0": "renamed1"}, output_path)
    assert output_path.with_name(f"{output_path.name}_data").exists()


def test_onnx_graphsurgeon_imports_with_onnx_121():
    ensure_onnx_helper_compatibility()
    import onnx_graphsurgeon as gs

    assert gs is not None
