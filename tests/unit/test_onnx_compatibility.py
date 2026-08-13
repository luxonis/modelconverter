"""Host-side unit tests for the external-data lookups.

Only these two helpers are covered here, because only they are new:
input staging asks them where a model keeps its weights so the companion
files can be copied into the cache alongside it. Miss one and the
container gets a model whose tensors are not there.

The wider compatibility helpers had a test module before the tests
refactor removed it; restoring that coverage belongs in its own change,
not in this one.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import set_external_data

from modelconverter.utils.onnx_compatibility import (
    get_external_data_paths,
    has_external_data,
)
from tests.helpers.onnx_factory import (
    single_io_onnx,
    weights_only_external_onnx,
)


def test_external_data_paths_find_the_companion_file(tmp_path: Path):
    model_path = tmp_path / "model.onnx"
    (external_data,) = weights_only_external_onnx(model_path)

    assert get_external_data_paths(model_path) == [external_data]
    assert has_external_data(model_path) is True


def test_has_external_data_is_false_without_external_data(tmp_path: Path):
    """Several callers branch on this to decide whether to re-save as
    external data at all, so an embedded model has to answer
    ``False``.
    """
    model_path = single_io_onnx(tmp_path / "model.onnx")

    assert get_external_data_paths(model_path) == []
    assert has_external_data(model_path) is False


def test_external_data_paths_cover_subgraph_initializers(tmp_path: Path):
    """A tensor reached only through a subgraph still names a file that
    has to travel with the model.

    Walking just the top-level graph -- the obvious implementation --
    would miss it, and the omission only surfaces later as a load
    failure inside the container.
    """
    model_path = tmp_path / "model.onnx"
    external_data = tmp_path / "subgraph_weights.bin"
    weights = np.asarray([1.0, 2.0], dtype=np.float32)
    external_data.write_bytes(weights.tobytes())

    tensor = numpy_helper.from_array(weights, name="weights")
    set_external_data(tensor, location=external_data.name)
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL

    # The tensor lives in the body of an `If`, not in the top-level graph.
    branch = helper.make_graph(
        [],
        "branch",
        [],
        [helper.make_tensor_value_info("weights", TensorProto.FLOAT, [2])],
        [tensor],
    )
    node = helper.make_node(
        "If", ["cond"], ["out"], then_branch=branch, else_branch=branch
    )
    graph = helper.make_graph(
        [node],
        "outer",
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
    )
    onnx.save(helper.make_model(graph), model_path)

    assert get_external_data_paths(model_path) == [external_data]


def test_external_data_paths_cover_attribute_tensors(tmp_path: Path):
    """A `Constant` keeps its value in a node attribute, and ONNX writes
    those out externally too (`convert_attribute=True`).

    Its own loader reads them back, so a companion file missed here
    becomes an "unable to open data file" once the container tries to
    convert a model whose weights never travelled with it.
    """
    model_path = tmp_path / "model.onnx"
    external_data = tmp_path / "constant_weights.bin"
    weights = np.asarray([1.0, 2.0], dtype=np.float32)
    external_data.write_bytes(weights.tobytes())

    tensor = numpy_helper.from_array(weights, name="const_value")
    set_external_data(tensor, location=external_data.name)
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL

    node = helper.make_node("Constant", [], ["out"], value=tensor)
    graph = helper.make_graph(
        [node],
        "outer",
        [],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2])],
    )
    onnx.save(helper.make_model(graph), model_path)

    assert get_external_data_paths(model_path) == [external_data]
