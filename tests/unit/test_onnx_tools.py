"""Host-side unit tests for the ONNX graph rewrites.

Only the Split-Concat fusion is covered here. The rest of the rewrites
need real models to say anything useful, so the conversion tests carry
them.
"""

from pathlib import Path

from modelconverter.utils.onnx_tools import ONNXModifier
from tests.helpers.onnx_factory import split_concat_onnx


def test_split_concat_fusion_leaves_a_terminal_concat_alone(tmp_path: Path):
    """A Concat that feeds nothing ends the forward walk for a ``Conv``.

    The walk recorded the missing successor before it tested for it, so
    the search then read ``op`` off ``None``. Every model whose Concat
    produced a graph output raised ``AttributeError`` here, which took
    down the whole conversion.
    """
    modifier = ONNXModifier(
        split_concat_onnx(tmp_path / "model.onnx"),
        tmp_path / "modified.onnx",
        skip_optimization=True,
        skip_constant_folding=True,
    )

    modifier._fuse_split_concat_to_conv()

    assert [node.op for node in modifier._onnx_gs.nodes] == [
        "Split",
        "Concat",
    ]
