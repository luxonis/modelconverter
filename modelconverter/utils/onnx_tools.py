"""ONNX graph utilities shared by the conversion backends.

Holds the passes modelconverter applies to an ONNX model before handing
it over to a platform's vendor toolchain: baking the configured input
normalization (channel reversal, mean subtraction and scaling) into the
graph, and simplifying, optimizing and fusing the graph with
``onnxsim``, ``onnxruntime`` and ``onnx_graphsurgeon``.
"""

import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final, cast

import numpy as np
import onnx
import onnxruntime as ort
from loguru import logger
from onnx import TensorProto, checker, helper
from onnxsim import simplify

from modelconverter.utils.config import (
    InputConfig,
    broadcast_preprocessing_values,
)
from modelconverter.utils.onnx_compatibility import (
    ensure_onnx_helper_compatibility,
    has_external_data,
    save_onnx_model,
)
from modelconverter.utils.preprocessing import normalization_required

from .exceptions import ONNXException, PreprocessingEmbeddingError

ensure_onnx_helper_compatibility()

# GraphSurgeon still imports helper conversion functions removed in ONNX 1.21.
# Patch them back in before importing GraphSurgeon.
import onnx_graphsurgeon as gs  # noqa: E402

FLOATING_TENSOR_TYPES: Final = frozenset(
    {
        TensorProto.FLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
        TensorProto.BFLOAT16,
    }
)


def get_opset_version(model: onnx.ModelProto) -> int:
    """Return the default domain opset version of an ONNX model.

    Args:
        model: Model whose ``opset_import`` is searched.

    Returns:
        Opset version declared for the default (empty) domain.

    Raises:
        ONNXException: If the model declares no default domain opset.

    """
    for imp in model.opset_import:
        if imp.domain == "":
            return imp.version
    raise ONNXException("No opset version found in the ONNX model.")


def onnx_attach_normalization_to_inputs(
    model_path: Path,
    save_path: Path,
    input_configs: dict[str, InputConfig],
    *,
    reverse_only: bool = False,
) -> Path:
    r"""Bake the input normalization into an ONNX model's graph.

    For every input that requires it, channel reversal (``Split`` and
    ``Concat``), mean subtraction (``Sub``) and scaling (``Mul`` by the
    reciprocal of the scale) nodes are inserted in front of the input,
    so that the graph itself computes

    .. math::

        y_c = (x_c - \mathrm{mean}_c) \cdot \frac{1}{\mathrm{scale}_c}

    for every channel :math:`c`. The resulting model is saved and
    validated with the ONNX checker. Mean and scale normalization supports
    any known channel count; color-channel reversal requires exactly three
    channels. A scalar mean or scale broadcasts to every channel.

    Args:
        model_path: Path to the source ONNX model.
        save_path: Path the modified model is saved to.
        input_configs: Input configurations keyed by input name.
        reverse_only: If ``True``, only the channel reversal is applied
            and the mean and scale values are ignored.

    Returns:
        ``save_path`` if any input required modification, otherwise the
        unmodified ``model_path``.

    Raises:
        ONNXException: If the preprocessing request or source model is invalid.
        PreprocessingEmbeddingError: If valid preprocessing cannot be embedded
            into this ONNX graph.

    """
    if not any(
        cfg.requires_input_preprocessing(reverse_only=reverse_only)
        for cfg in input_configs.values()
    ):
        logger.info(
            "No ONNX input normalization changes requested; using original model."
        )
        return model_path

    try:
        checker.check_model(str(model_path))
    except checker.ValidationError as e:
        raise ONNXException(
            f"The source ONNX model failed validation: {e}"
        ) from e

    model = onnx.load(str(model_path))
    model_has_external_data = has_external_data(model_path)

    graph = model.graph
    input_names = [input_tensor.name for input_tensor in graph.input]
    output_names = {output_tensor.name for output_tensor in graph.output}
    if not all(name in input_names for name in input_configs):
        raise ONNXException(
            "You either used an invalid input name, or you're attempting "
            "to use a hidden network node as an input. This is not supported "
            "in combination with input modifications (mean, scale, etc.). "
            "Either use an actual input name, or modify your network."
        )

    new_initializers = []
    for input_tensor in graph.input:
        input_name = input_tensor.name
        cfg = input_configs.get(input_name)
        if cfg is None or not cfg.requires_input_preprocessing(
            reverse_only=reverse_only
        ):
            continue

        if input_name in output_names:
            raise PreprocessingEmbeddingError(
                f"Cannot embed preprocessing for input '{input_name}' because "
                "the same tensor is exposed directly as a graph output."
            )

        new_nodes, last_output = _preprocessing_nodes(
            model,
            input_tensor,
            cfg,
            new_initializers,
            reverse_only=reverse_only,
        )
        _insert_before_consumers(graph, input_name, last_output, new_nodes)

    graph.initializer.extend(new_initializers)

    save_onnx_model(
        model,
        save_path,
        save_as_external_data=model_has_external_data,
        location=f"{save_path.name}_data",
    )

    try:
        checker.check_model(str(save_path))
    except checker.ValidationError as e:
        raise PreprocessingEmbeddingError(
            "The ONNX graph produced while embedding preprocessing failed "
            f"validation: {e}"
        ) from e

    return save_path


def _preprocessing_nodes(
    model: onnx.ModelProto,
    input_tensor: onnx.ValueInfoProto,
    cfg: InputConfig,
    initializers: list[onnx.TensorProto],
    *,
    reverse_only: bool,
) -> tuple[list[onnx.NodeProto], str]:
    """Build the nodes that preprocess one input of the graph.

    The nodes reverse the channels, subtract the mean and multiply by
    the reciprocal of the scale, in this order, each only when needed.
    The constants they use are appended to ``initializers``.

    Returns:
        The new nodes and the name of the preprocessed tensor.
    """
    input_name = input_tensor.name
    input_dtype = input_tensor.type.tensor_type.elem_type
    n_channels, mean_values, scale_values = _normalization_values(
        cfg, input_name, input_dtype, reverse_only=reverse_only
    )
    layout = cfg.layout
    assert layout is not None

    nodes = []
    last_output = input_name
    if cfg.encoding_mismatch:
        nodes += _reverse_channel_nodes(
            model, input_name, layout, n_channels, initializers
        )
        last_output = nodes[-1].output[0]

    if mean_values is not None and any(v != 0 for v in mean_values):
        node, tensor = _channel_op(
            "Sub",
            f"sub_out_{input_name}",
            f"mean_{input_name}",
            last_output,
            mean_values,
            layout=layout,
            dtype=input_dtype,
        )
        nodes.append(node)
        initializers.append(tensor)
        last_output = node.output[0]

    if scale_values is not None and any(v != 1 for v in scale_values):
        node, tensor = _channel_op(
            "Mul",
            f"div_out_{input_name}",
            f"scale_{input_name}",
            last_output,
            [1 / v for v in scale_values],
            layout=layout,
            dtype=input_dtype,
        )
        nodes.append(node)
        initializers.append(tensor)
        last_output = node.output[0]

    return nodes, last_output


def _normalization_values(
    cfg: InputConfig, input_name: str, input_dtype: int, *, reverse_only: bool
) -> tuple[int, list[float] | None, list[float] | None]:
    """Check that the preprocessing of an input can go into the graph.

    Returns:
        The channel count, and the mean and scale values broadcast to
        it. With ``reverse_only``, both value lists are ``None``.
    """
    try:
        n_channels = cfg.validate_preprocessing(reverse_only=reverse_only)
    except ValueError as e:
        raise ONNXException(str(e)) from e

    if cfg.layout not in ["NCHW", "NHWC"]:
        raise PreprocessingEmbeddingError(
            f"Cannot embed preprocessing for input '{input_name}' with "
            f"layout '{cfg.layout}'; only 'NCHW' and 'NHWC' are supported."
        )
    if reverse_only:
        return n_channels, None, None

    mean_values = (
        None
        if cfg.mean_values is None
        else broadcast_preprocessing_values(cfg.mean_values, n_channels)
    )
    scale_values = (
        None
        if cfg.scale_values is None
        else broadcast_preprocessing_values(cfg.scale_values, n_channels)
    )
    if (
        normalization_required(mean_values, scale_values)
        and input_dtype not in FLOATING_TENSOR_TYPES
    ):
        dtype_name = TensorProto.DataType.Name(input_dtype)
        raise PreprocessingEmbeddingError(
            f"Cannot embed mean/scale preprocessing for input "
            f"'{input_name}' with ONNX data type '{dtype_name}'; "
            "normalization requires a floating-point input."
        )
    return n_channels, mean_values, scale_values


def _reverse_channel_nodes(
    model: onnx.ModelProto,
    input_name: str,
    layout: str,
    n_channels: int,
    initializers: list[onnx.TensorProto],
) -> list[onnx.NodeProto]:
    """Build the ``Split`` and ``Concat`` nodes that reverse the channels.

    From opset 13, the split lengths are an input, which is appended to
    ``initializers``.
    """
    try:
        opset = get_opset_version(model)
    except ONNXException as e:
        raise PreprocessingEmbeddingError(
            f"Cannot embed channel reversal for input '{input_name}': {e}"
        ) from e
    split_names = [f"split_{i}_{input_name}" for i in range(3)]
    axis = 1 if layout == "NCHW" else 3

    if opset < 13:
        split_node = helper.make_node(
            "Split",
            inputs=[input_name],
            outputs=split_names,
            axis=axis,
            split=[1] * n_channels,
            name=f"split_{input_name}",
        )
    else:
        split_lengths_name = f"split_{input_name}_lengths"
        initializers.append(
            helper.make_tensor(
                name=split_lengths_name,
                data_type=TensorProto.INT64,
                dims=[n_channels],
                vals=[1] * n_channels,
            )
        )
        split_node = helper.make_node(
            "Split",
            inputs=[input_name, split_lengths_name],
            outputs=split_names,
            axis=axis,
            name=f"split_{input_name}",
        )

    concat_node = helper.make_node(
        "Concat",
        inputs=split_names[::-1],
        outputs=[f"normalized_{input_name}"],
        axis=axis,
        name=f"concat_{input_name}",
    )
    return [split_node, concat_node]


def _channel_op(
    op: str,
    output_name: str,
    constant_name: str,
    input_name: str,
    values: list[float],
    *,
    layout: str,
    dtype: int,
) -> tuple[onnx.NodeProto, onnx.TensorProto]:
    """Build a node that applies per-channel ``values`` with ``op``.

    Returns:
        The node, named after its output, and its constant.
    """
    node = helper.make_node(
        op,
        inputs=[input_name, constant_name],
        outputs=[output_name],
        name=output_name,
    )
    shape = (
        [1, len(values), 1, 1] if layout == "NCHW" else [1, 1, 1, len(values)]
    )
    return node, helper.make_tensor(constant_name, dtype, shape, values)


def _insert_before_consumers(
    graph: onnx.GraphProto,
    input_name: str,
    last_output: str,
    new_nodes: list[onnx.NodeProto],
) -> None:
    """Feed the consumers of an input from ``last_output``.

    The new nodes go in front of the first node that reads
    ``last_output``.
    """
    for node in graph.node:
        new_inputs = [
            last_output if inp == input_name else inp for inp in node.input
        ]
        del node.input[:]
        node.input.extend(new_inputs)

    idx = next(
        (i for i, node in enumerate(graph.node) if last_output in node.input),
        0,
    )
    nodes_as_list = list(graph.node)
    nodes_as_list[idx:idx] = new_nodes
    del graph.node[:]
    graph.node.extend(nodes_as_list)


_ORT_INPUT_TYPES: Final[dict[str, type[np.generic] | str]] = {
    "tensor(float64)": np.float64,
    "tensor(float32)": np.float32,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(bool)": "bool",
}

_BN_FUSION_PATTERNS: Final = (
    ("Conv", "Add", "Mul"),
    ("Conv", "Mul", "Add"),
    ("Conv", "Mul"),
    ("Conv", "Add"),
)


class ONNXModifier:
    """ONNX model modifier class to optimize and modify the ONNX model.

    Attributes:
        model_path: Path to the base ONNX model.
        output_path: Path to save the modified ONNX model.

    """

    def __init__(
        self,
        model_path: Path,
        output_path: Path,
        skip_optimization: bool = False,
        skip_constant_folding: bool = False,
    ) -> None:
        """Load the ONNX model and prepare it for modification.

        Args:
            model_path: Path to the base ONNX model.
            output_path: Path to save the modified ONNX model to.
            skip_optimization: Whether to skip the graph optimization
                performed while loading and exporting the model.
            skip_constant_folding: Whether to skip constant folding
                during simplification.

        """
        self.model_path = model_path
        self._has_external_data = has_external_data(model_path)
        self.output_path = output_path
        self._skip_optimization = skip_optimization
        self._skip_constant_folding = skip_constant_folding
        self._load_onnx()
        self._prev_onnx_model = self._onnx_model
        self._prev_onnx_gs = self._onnx_gs

    def modify_onnx(
        self,
        substitute_sub_with_add: bool = True,
        substitute_div_with_mul: bool = True,
        fuse_add_mul_to_bn: bool = True,
        fuse_comb_add_mul_to_conv: bool = True,
        fuse_single_add_mul_to_conv: bool = True,
        fuse_split_concat_to_conv: bool = True,
    ) -> bool:
        """Modify the ONNX model by applying a series of optimizations.

        Each flag enables one step. A step that changes the outputs of
        the model is reverted.

        Args:
            substitute_sub_with_add: Whether to substitute ``Sub`` nodes
                with ``Add`` nodes.
            substitute_div_with_mul: Whether to substitute ``Div`` nodes
                with ``Mul`` nodes.
            fuse_add_mul_to_bn: Whether to fuse ``Add`` and ``Mul`` nodes
                into ``BatchNormalization`` nodes.
            fuse_comb_add_mul_to_conv: Whether to fuse combinations of
                ``Add`` and ``Mul`` nodes into ``Conv`` nodes.
            fuse_single_add_mul_to_conv: Whether to fuse single ``Add``
                and ``Mul`` nodes into ``Conv`` nodes.
            fuse_split_concat_to_conv: Whether to fuse ``Split`` and
                ``Concat`` nodes into ``Conv`` nodes.

        Returns:
            ``True`` if the model was modified and exported,
            ``False`` otherwise.

        """
        if self._has_dynamic_shape:
            logger.warning(
                "Identified dynamic input shape, skipping model modifications..."
            )
            return False

        if substitute_div_with_mul:
            self._apply_optimization_step(
                "Substitute Div -> Mul nodes",
                lambda: self._substitute_node_by_type(
                    source_node="Div", target_node="Mul"
                ),
            )
        if substitute_sub_with_add:
            self._apply_optimization_step(
                "Substitute Sub -> Add nodes",
                lambda: self._substitute_node_by_type(
                    source_node="Sub", target_node="Add"
                ),
            )
        if fuse_add_mul_to_bn:
            self._apply_optimization_step(
                "Fuse Add and Mul nodes to BatchNormalization nodes",
                self._fuse_add_mul_to_bn,
            )
        if fuse_comb_add_mul_to_conv:
            self._apply_optimization_step(
                "Fuse Add and Mul nodes to Conv nodes (combined)",
                self._fuse_comb_add_mul_to_conv,
            )
        if fuse_single_add_mul_to_conv:
            self._apply_optimization_step(
                "Fuse Add and Mul nodes to Conv nodes (single)",
                self._fuse_single_add_mul_to_conv,
            )
        if fuse_split_concat_to_conv:
            self._apply_optimization_step(
                "Fuse Split and Concat nodes to Conv nodes",
                self._fuse_split_concat_to_conv,
            )

        try:
            self._export_onnx()
        except Exception as e:
            logger.error(f"Failed to modify the ONNX model: {e}")
            return False

        return True

    def compare_outputs(self, from_modelproto: bool = False) -> bool:
        """Compare the outputs of two ONNX models.

        Args:
            from_modelproto: If ``True``, compare the in-memory model
                against its previous state instead of comparing the
                model files on disk.

        Returns:
            ``True`` if the outputs of both models match,
            ``False`` otherwise.

        """
        import onnxruntime as ort

        ort.set_default_logger_severity(3)

        if from_modelproto:
            onnx_model_1 = self._prev_onnx_model.SerializeToString()
            onnx_model_2 = self._onnx_model.SerializeToString()
        else:
            onnx_model_1 = self.model_path.as_posix()
            onnx_model_2 = self.output_path.as_posix()

        ort_session_1 = ort.InferenceSession(onnx_model_1)
        ort_session_2 = ort.InferenceSession(onnx_model_2)

        inputs = {
            input.name: np.random.rand(*input.shape).astype(
                _ORT_INPUT_TYPES[input.type]
            )
            for input in ort_session_1.get_inputs()
        }

        outputs_1 = ort_session_1.run(None, inputs)

        outputs_2 = ort_session_2.run(None, inputs)

        equal_outputs = True
        for out1, out2 in zip(outputs_1, outputs_2, strict=True):
            # A sequence or map output has nothing to compare numerically.
            if not isinstance(out1, np.ndarray) or not isinstance(
                out2, np.ndarray
            ):
                raise ONNXException(
                    "Can only compare models whose outputs are all tensors."
                )
            equal_outputs = equal_outputs and np.allclose(
                out1, out2, rtol=5e-3, atol=5e-3
            )

        return equal_outputs

    def _load_onnx(self) -> None:
        """Load the ONNX model and store it as ``onnx.ModelProto`` and
        ``onnx_graphsurgeon`` graph.
        """
        logger.info(f"Loading model: {self.model_path.stem}")

        try:
            self._onnx_model, _ = simplify(
                self.model_path.as_posix(),
                perform_optimization=not self._skip_optimization,
                skip_constant_folding=self._skip_constant_folding,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load and simplify ONNX model: {self.model_path.stem} with error: {e}\nLoading without simplification."
            )
            self._onnx_model = onnx.load(self.model_path.as_posix())
        onnx.checker.check_model(self._onnx_model)

        self._dtype = helper.tensor_dtype_to_np_dtype(
            self._onnx_model.graph.input[0].type.tensor_type.elem_type
        )
        self._input_shape = [
            dim.dim_value
            for dim in self._onnx_model.graph.input[
                0
            ].type.tensor_type.shape.dim
        ]
        self._has_dynamic_shape = any(
            dim == 0 or dim is None for dim in self._input_shape
        )

        self._onnx_gs = gs.import_onnx(self._onnx_model)

    def _optimize_onnx(self) -> None:
        """Optimize and simplify the ONNX model's graph."""
        self._onnx_model.ir_version = min(self._onnx_model.ir_version, 10)

        if self._skip_optimization:
            optimized_onnx_model = self._onnx_model
        else:
            with tempfile.NamedTemporaryFile(
                suffix=".onnx", delete=True
            ) as tmp_file:
                onnx.save(self._onnx_model, tmp_file.name)

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                )
                sess_options.optimized_model_filepath = tmp_file.name.replace(
                    ".onnx", "_optimized.onnx"
                )

                _ = ort.InferenceSession(tmp_file.name, sess_options)

                optimized_onnx_model = onnx.load(
                    sess_options.optimized_model_filepath
                )

                Path(sess_options.optimized_model_filepath).unlink()

        try:
            optimized_onnx_model, _ = simplify(
                optimized_onnx_model,
                perform_optimization=False,
                skip_constant_folding=self._skip_constant_folding,
            )
        except Exception as e:
            logger.warning(
                f"Failed to simplify optimized ONNX model with error: {e}\nProceeding without simplification."
            )

        if self._has_external_data:
            with tempfile.NamedTemporaryFile(
                delete=True, suffix=".onnx"
            ) as tmp_onnx_file:
                save_onnx_model(
                    optimized_onnx_model,
                    tmp_onnx_file.name,
                    save_as_external_data=True,
                    location=f"{Path(tmp_onnx_file.name).name}_data",
                )
                onnx.checker.check_model(tmp_onnx_file.name)
        else:
            onnx.checker.check_model(optimized_onnx_model)

        self._onnx_model, self._onnx_gs = (
            optimized_onnx_model,
            gs.import_onnx(optimized_onnx_model),
        )

    def _export_onnx(self) -> None:
        """Export the modified ONNX model to the output path."""
        self._optimize_onnx()

        self._onnx_model.ir_version = min(self._onnx_model.ir_version, 10)

        save_onnx_model(
            self._onnx_model,
            self.output_path,
            save_as_external_data=self._has_external_data,
            location=f"{self.output_path.name}_data",
        )
        onnx.checker.check_model(str(self.output_path))

    def _add_outputs(self, output_names: list[str]) -> None:
        """Expose the named tensors as outputs of the ONNX model.

        Args:
            output_names: Names of the output tensors to add to the
                model's outputs.

        """
        graph_outputs = _output_names(self._onnx_gs)
        for name, tensor in self._onnx_gs.tensors().items():
            if name in output_names and name not in graph_outputs:
                self._onnx_gs.outputs.append(tensor)
        self._onnx_model = gs.export_onnx(self._onnx_gs)

    def _get_constant_map(self, graph: gs.Graph) -> dict[str, np.ndarray]:
        """Extract constant tensors from the GraphSurgeon graph.

        Args:
            graph: Graph whose ``gs.Constant`` tensors are collected.

        Returns:
            Constant tensor map with tensor name as key and tensor value
            as value.

        """
        return {
            tensor.name: tensor.values
            for tensor in graph.tensors().values()
            if isinstance(tensor, gs.Constant)
        }

    @staticmethod
    def _get_constant_value(
        node: gs.Node, constant_map: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, int] | None:
        """Return the constant value of a node if it is a constant node.

        Args:
            node: Node whose inputs are searched for a constant.
            constant_map: Constant tensor map with tensor name as key
                and tensor value as value.

        Returns:
            Constant tensor value and index, or ``None`` if the node has
            no constant input.

        """
        for idx, input in enumerate(node.inputs):
            if input.name in constant_map:
                return (constant_map[input.name], idx)

        return None

    @staticmethod
    def _get_variable_input(node: gs.Node) -> tuple[gs.Variable, int] | None:
        """Return the variable input of a node.

        Args:
            node: Node whose inputs are searched for a variable.

        Returns:
            Variable input and index, or ``None`` if the node has no
            variable input.

        """
        for idx, input in enumerate(node.inputs):
            if isinstance(input, gs.Variable):
                return (input, idx)

        return None

    def _graph_cleanup(
        self,
        nodes_to_add: list[gs.Node],
        nodes_to_remove: list[gs.Node],
        connections_to_fix: list[tuple[gs.Variable, gs.Variable]],
    ) -> None:
        """Clean up the graph by adding new nodes, removing old nodes,
        and fixing connections.

        Args:
            nodes_to_add: List of nodes to add to the graph.
            nodes_to_remove: List of nodes to remove from the graph.
            connections_to_fix: List of connections to fix in the graph.

        """
        # GraphSurgeon declares the nodes as an immutable sequence, but
        # the graph keeps them in a list that this surgery edits in place.
        nodes = self._onnx_gs.nodes
        assert isinstance(nodes, list)
        for node in nodes_to_add:
            nodes.append(node)

        for old_input, new_input in connections_to_fix:
            for node in nodes:
                for idx, input in enumerate(node.inputs):
                    if input == old_input:
                        node.inputs[idx] = new_input

        for node in nodes_to_remove:
            nodes.remove(node)

        self._onnx_gs.cleanup(
            remove_unused_node_outputs=True, remove_unused_graph_inputs=True
        ).toposort()

    def _substitute_node_by_type(
        self, source_node: str, target_node: str
    ) -> None:
        """Substitute a source node of a particular type with a target
        node of a different type.

        Currently, only ``Sub -> Add`` and ``Div -> Mul`` substitutions
        are allowed.

        Args:
            source_node: Source node type to substitute.
            target_node: Target node type to substitute with.

        Raises:
            ValueError: If the source and target node types do not form
                an allowed substitution pair.

        """
        if source_node not in ["Sub", "Div"] or target_node not in [
            "Add",
            "Mul",
        ]:
            raise ValueError(
                "Invalid source or target node type. Valid source types: Sub, Div. Valid target types: Add, Mul."
            )

        if (source_node, target_node) not in {("Sub", "Add"), ("Div", "Mul")}:
            raise ValueError(
                "Invalid substitution. Available substitutions: Sub -> Add, Div -> Mul"
            )

        constant_map = self._get_constant_map(self._onnx_gs)

        nodes_to_add = []
        nodes_to_remove = []
        connections_to_fix = []

        for node in self._onnx_gs.nodes:
            if node.op != source_node:
                continue
            if self._feeds_erf(node):
                logger.warning(
                    f"Skipping the `{source_node} -> {target_node}` "
                    f"substitution optimization for node '{node.name}' "
                    f"with op '{node.op}' because it is followed by an "
                    "'Erf' node."
                )
                continue
            constant = self._get_constant_value(node, constant_map)
            if constant is None:
                continue
            new_node = _substitute_node(
                node, target_node, const_idx=constant[1]
            )
            if new_node is None:
                continue

            nodes_to_add.append(new_node)
            connections_to_fix.append((node.outputs[0], new_node.outputs[0]))
            self._take_over_graph_output(node, new_node)
            nodes_to_remove.append(node)

        self._commit_graph_edits(
            nodes_to_add,
            nodes_to_remove,
            connections_to_fix,
            nothing_found="No applicable Sub-Add or Div-Mul pattern found for substitution.",
        )

    def _feeds_erf(self, node: gs.Node) -> bool:
        """Tell whether an ``Erf`` node reads the output of ``node``.

        Such a node is not substituted, because SNPE 2.32.6 fails to
        convert the result.
        """
        return any(
            n.op == "Erf"
            for n in self._onnx_gs.nodes
            if node.outputs[0] in n.inputs
        )

    def _take_over_graph_output(
        self, node: gs.Node, new_node: gs.Node
    ) -> None:
        """Make ``new_node`` produce the graph output of ``node``.

        The output keeps its name. Nothing changes when ``node`` does
        not produce a graph output.
        """
        output = node.outputs[0]
        for i, graph_output_name in enumerate(_output_names(self._onnx_gs)):
            if graph_output_name == output.name:
                new_output_var = gs.Variable(
                    name=output.name, dtype=output.dtype, shape=output.shape
                )
                new_node.outputs[0] = new_output_var
                self._onnx_gs.outputs[i] = new_output_var
                return

    def _fuse_add_mul_to_bn(self) -> None:
        """Fuse Add/Sub and Mul nodes that come immediately after a Conv
        node into a BatchNormalization node.

        The fusion patterns considered are:
        1. Conv -> Add -> Mul
        2. Conv -> Mul -> Add
        3. Conv -> Mul
        4. Conv -> Add
        """
        constant_map = self._get_constant_map(self._onnx_gs)

        nodes_to_add = []
        nodes_to_remove = []
        connections_to_fix = []

        sequences = _longest_sequences(
            self._find_sequences(_BN_FUSION_PATTERNS)
        )
        for sequence in sequences:
            conv_node = sequence[0]
            folded = self._fold_sequence(sequence, constant_map)
            if folded is None or len(conv_node.outputs[0].outputs) > 1:
                continue
            scale, bias = folded

            bn_node = _batch_norm_node(
                f"BatchNorm_{conv_node.name.replace('/', '', 1)}",
                conv_node.outputs[0],
                scale,
                bias,
                dtype=self._dtype,
            )
            nodes_to_add.append(bn_node)
            connections_to_fix.append(
                (sequence[-1].outputs[0], bn_node.outputs[0])
            )
            nodes_to_remove.extend(sequence[1:])

        self._commit_graph_edits(
            nodes_to_add,
            nodes_to_remove,
            connections_to_fix,
            nothing_found="No applicable Conv-Add-Mul pattern found for batch normalization fusion.",
        )

    def _find_sequences(
        self, patterns: Iterable[tuple[str, ...]]
    ) -> list[list[gs.Node]]:
        """Find the node chains whose operations follow a pattern.

        Each node of a chain reads the first output of the node before.
        """
        sequences = []
        for pattern in patterns:
            for node in self._onnx_gs.nodes:
                if node.op != pattern[0]:
                    continue
                sequence = self._follow_ops(node, pattern[1:])
                if len(sequence) == len(pattern):
                    sequences.append(sequence)
        return sequences

    def _follow_ops(
        self, node: gs.Node, ops: tuple[str, ...]
    ) -> list[gs.Node]:
        """Follow ``node`` through consumers of the operations in ``ops``.

        Returns:
            ``node`` and the consumers found, up to the first operation
            without a consumer.
        """
        sequence = [node]
        for op_type in ops:
            next_nodes = [
                n
                for n in self._onnx_gs.nodes
                if n.inputs
                and sequence[-1].outputs[0] in n.inputs
                and n.op == op_type
            ]
            if not next_nodes:
                break
            sequence.append(next_nodes[0])
        return sequence

    def _fold_sequence(
        self, sequence: list[gs.Node], constant_map: dict[str, np.ndarray]
    ) -> tuple[float | np.ndarray, float | np.ndarray] | None:
        """Fold the constants after the leading Conv into a scale and a
        bias.

        Returns:
            The scale and the bias, or ``None`` when a node after the
            Conv has no constant input.
        """
        scale, bias = 1.0, 0.0
        for seq_node in sequence[1:]:
            constant = self._get_constant_value(seq_node, constant_map)
            if constant is None:
                return None
            constant_val, _ = constant
            if seq_node.op == "Add":
                bias += constant_val
            else:
                scale *= constant_val
        return scale, bias

    def _fuse_single_add_mul_to_conv(self) -> None:
        """Fuse Add and Mul nodes that precede a Conv node directly into
        the Conv node.
        """
        nodes_to_remove = []
        connections_to_fix = []

        constant_map = self._get_constant_map(self._onnx_gs)

        for node in self._onnx_gs.nodes:
            if node.op == "Mul":
                fused = self._fuse_mul_into_next_conv(node, constant_map)
            elif node.op == "Add":
                fused = self._fuse_add_into_next_conv(node, constant_map)
            else:
                continue
            if fused:
                nodes_to_remove.append(node)
                connections_to_fix.append((node.outputs[0], node.inputs[0]))

        self._commit_graph_edits(
            [],
            nodes_to_remove,
            connections_to_fix,
            nothing_found="No applicable Add-Mul-Conv pattern found for fusion.",
        )

    def _fuse_mul_into_next_conv(
        self, mul_node: gs.Node, constant_map: dict[str, np.ndarray]
    ) -> bool:
        """Fold a constant ``Mul`` into the weights of the Conv after it.

        Returns:
            Whether the Conv took over the ``Mul``.
        """
        if len(mul_node.outputs[0].outputs) > 1:
            return False
        conv_node = _next_node(mul_node, "Conv")
        if conv_node is None:
            return False
        constant = self._get_constant_value(mul_node, constant_map)
        if constant is None:
            return False
        _scale_conv_weights(conv_node, constant[0])
        return True

    def _fuse_add_into_next_conv(
        self, add_node: gs.Node, constant_map: dict[str, np.ndarray]
    ) -> bool:
        """Fold a constant ``Add`` into the bias of the Conv after it.

        A Conv with padding does not take the ``Add``, because the
        padded border would not get the added value.

        Returns:
            Whether the Conv took over the ``Add``.
        """
        if len(add_node.outputs[0].outputs) > 1:
            return False
        conv_node = _next_node(add_node, "Conv")
        if conv_node is None or _has_padding(conv_node):
            return False
        constant = self._get_constant_value(add_node, constant_map)
        if constant is None:
            return False
        _shift_conv_bias(conv_node, constant[0], conv_node.inputs[1])
        return True

    def _fuse_comb_add_mul_to_conv(self) -> None:
        """Fuse combinations of Add and Mul nodes preceding a Conv node
        directly into the Conv node itself.

        The fusion patterns considered are:
        1. Add -> Mul -> Conv
        2. Mul -> Add -> Conv
        """
        nodes_to_remove = []
        connections_to_fix = []

        constant_map = self._get_constant_map(self._onnx_gs)

        for node in self._onnx_gs.nodes:
            if node.op == "Mul":
                second = self._fuse_mul_add_into_conv(node, constant_map)
            elif node.op == "Add":
                second = self._fuse_add_mul_into_conv(node, constant_map)
            else:
                continue
            if second is None:
                continue

            variable = self._get_variable_input(node)
            if variable is None:
                continue
            nodes_to_remove += [node, second]
            connections_to_fix.append(
                (second.outputs[0], node.inputs[variable[1]])
            )

        self._commit_graph_edits(
            [],
            nodes_to_remove,
            connections_to_fix,
            nothing_found="No applicable Add-Mul-Conv pattern found for fusion.",
        )

    def _fuse_mul_add_into_conv(
        self, mul_node: gs.Node, constant_map: dict[str, np.ndarray]
    ) -> gs.Node | None:
        """Fold a constant ``Mul -> Add`` into the Conv after it.

        Returns:
            The ``Add`` node, or ``None`` when the chain does not fit.
        """
        found = _conv_after(mul_node, "Add")
        if found is None:
            return None
        add_node, conv_node = found
        mul_constant = self._get_constant_value(mul_node, constant_map)
        if mul_constant is None:
            return None
        add_constant = self._get_constant_value(add_node, constant_map)
        if add_constant is None:
            return None

        conv_weights = conv_node.inputs[1]
        _scale_conv_weights(conv_node, mul_constant[0])
        _shift_conv_bias(conv_node, add_constant[0], conv_weights)
        return add_node

    def _fuse_add_mul_into_conv(
        self, add_node: gs.Node, constant_map: dict[str, np.ndarray]
    ) -> gs.Node | None:
        """Fold a constant ``Add -> Mul`` into the Conv after it.

        Returns:
            The ``Mul`` node, or ``None`` when the chain does not fit.
        """
        found = _conv_after(add_node, "Mul")
        if found is None:
            return None
        mul_node, conv_node = found
        add_constant = self._get_constant_value(add_node, constant_map)
        if add_constant is None:
            return None
        mul_constant = self._get_constant_value(mul_node, constant_map)
        if mul_constant is None:
            return None

        add_value, _ = add_constant
        mul_value, _ = mul_constant
        add_value *= mul_value
        conv_weights = conv_node.inputs[1]
        _shift_conv_bias(conv_node, add_value, conv_weights)
        _scale_conv_weights(conv_node, mul_value)
        return mul_node

    def _fuse_split_concat_to_conv(self) -> None:
        """Fuse Split and Concat nodes that come before a Conv node into
        the Conv node.

        If any intermediate nodes have channel dimensions, the order of
        the channels is reversed.
        """
        nodes_to_remove = []
        connections_to_fix = []

        for node in self._onnx_gs.nodes:
            if node.op == "Conv":
                break
            if node.op != "Split":
                continue

            found = _split_concat_conv(node)
            if found is None:
                continue
            concat_node, intermediate_nodes = found
            if not self._reverse_conv_channels(
                node, concat_node, intermediate_nodes
            ):
                break

            nodes_to_remove += [node, concat_node]
            connections_to_fix.append((concat_node.outputs[0], node.inputs[0]))
            break

        self._commit_graph_edits(
            [],
            nodes_to_remove,
            connections_to_fix,
            nothing_found="No applicable Split-Conv-Concat pattern found for fusion.",
        )

    def _reverse_conv_channels(
        self,
        split_node: gs.Node,
        concat_node: gs.Node,
        intermediate_nodes: list[gs.Node],
    ) -> bool:
        """Reverse the channels of the Conv weights and the constants before.

        The last of ``intermediate_nodes`` is the Conv.

        Returns:
            ``False`` when the Conv does not have 1 or 3 channels on the
            axis of the split.
        """
        conv_weights = intermediate_nodes[-1].inputs[1]

        if split_node.attrs["axis"] != concat_node.attrs["axis"]:
            raise ValueError(
                f"Split and Concat axis mismatch: {split_node.attrs['axis']} != {concat_node.attrs['axis']}"
            )

        channels_axis = split_node.attrs["axis"]
        if not isinstance(channels_axis, int):
            raise TypeError(
                f"Split node axis must be an integer, got: {channels_axis}"
            )
        if conv_weights.shape[channels_axis] not in [1, 3]:
            return False

        for inter_node in intermediate_nodes[:-1]:
            self._flip_constant(inter_node, channels_axis, conv_weights)

        conv_weights.values = np.flip(conv_weights.values, axis=channels_axis)
        return True

    def _flip_constant(
        self, node: gs.Node, channels_axis: int, conv_weights: gs.Constant
    ) -> None:
        """Reverse the channels of the constant input of a node.

        A node without a constant, or with a 1D constant, keeps it.
        """
        constant = self._get_constant_value(
            node, self._get_constant_map(self._onnx_gs)
        )
        if constant is None:
            return
        constant_value, constant_idx = constant
        if constant_value.ndim == 1:
            return

        if constant_value.shape[channels_axis] != conv_weights.values.shape[1]:
            logger.warning(
                f"Spatial dimensions mismatch between Conv and intermediate node {node.name}: {constant_value.shape[channels_axis]} != {conv_weights.values.shape[1]}, discarding this step."
            )

        node.inputs[constant_idx].values = np.flip(
            constant_value, axis=channels_axis
        )

    def _commit_graph_edits(
        self,
        nodes_to_add: list[gs.Node],
        nodes_to_remove: list[gs.Node],
        connections_to_fix: list[tuple[gs.Variable, gs.Variable]],
        *,
        nothing_found: str,
    ) -> None:
        """Apply the edits of an optimization step and optimize the result.

        A step without edits only logs ``nothing_found``.
        """
        if not any([nodes_to_add, nodes_to_remove, connections_to_fix]):
            logger.warning(nothing_found)
            return

        self._graph_cleanup(nodes_to_add, nodes_to_remove, connections_to_fix)
        self._onnx_model = gs.export_onnx(self._onnx_gs)

        self._optimize_onnx()

    def _revert_changes(self) -> None:
        """Revert the ONNX model to its previous state."""
        self._onnx_model = self._prev_onnx_model
        self._onnx_gs = self._prev_onnx_gs

    def _apply_optimization_step(
        self, step_name: str, optimization_func: Callable
    ) -> None:
        """Apply a single optimization step to the ONNX model.

        If the step fails or changes the model outputs, the model is
        reverted to its previous state.

        Args:
            step_name: Name of the step, used in the debug log.
            optimization_func: Optimization function to apply.

        """
        logger.debug(f"Attempting: {step_name}...")
        try:
            optimization_func()
            if not self.compare_outputs(from_modelproto=True):
                logger.warning(
                    f"Failed: {step_name} due to output mismatch, reverting changes..."
                )
                self._revert_changes()
        except Exception as e:
            logger.warning(
                f"Failed: {step_name} with error: {e}, reverting changes..."
            )
            self._revert_changes()


def _substitute_node(
    node: gs.Node, target_node: str, *, const_idx: int
) -> gs.Node | None:
    """Build the ``Add`` or ``Mul`` node that replaces a ``Sub`` or ``Div``.

    The ``Add`` adds the negated constant, and the ``Mul`` multiplies by
    its reciprocal.

    Returns:
        ``None`` when the constant is the first input, or when the
        constant of a ``Div`` is not a floating-point tensor.
    """
    if const_idx == 0:
        return None

    first_input = node.inputs[0]
    second_input = node.inputs[const_idx]
    if target_node == "Add":
        new_value = -second_input.values
    elif second_input.dtype in [np.float16, np.float32, np.float64]:
        new_value = 1.0 / second_input.values
    else:
        return None
    return gs.Node(
        op=target_node,
        inputs=[
            first_input,
            gs.Constant(
                name=f"{node.name}_{second_input.name}/Substitute",
                values=np.array(new_value, dtype=second_input.dtype),
            ),
        ],
        outputs=[gs.Variable(name=f"{node.name}/{target_node}_output")],
        name=f"{node.name}/To_{target_node}",
    )


def _longest_sequences(
    sequences: list[list[gs.Node]],
) -> list[list[gs.Node]]:
    """Drop each node chain that a longer chain contains."""
    return [
        seq
        for seq in sequences
        if not any(
            all(node in longer_seq for node in seq)
            and len(seq) < len(longer_seq)
            for longer_seq in sequences
        )
    ]


def _batch_norm_node(
    name: str,
    input_tensor: gs.Variable,
    scale: float | np.ndarray,
    bias: float | np.ndarray,
    *,
    dtype: np.dtype,
) -> gs.Node:
    """Build a ``BatchNormalization`` node that computes ``x * scale + bias``.

    The mean is zero and the variance is one on every channel.
    """
    assert input_tensor.shape is not None
    conv_channels = int(input_tensor.shape[1])
    scale_values = np.array([scale] * conv_channels, dtype=dtype).squeeze()
    bias_values = np.array([bias] * conv_channels, dtype=dtype).squeeze()
    return gs.Node(
        op="BatchNormalization",
        inputs=[
            input_tensor,
            gs.Constant(name=f"{name}_scale", values=scale_values),
            gs.Constant(name=f"{name}_bias", values=bias_values),
            gs.Constant(
                name=f"{name}_mean", values=np.zeros_like(scale_values)
            ),
            gs.Constant(name=f"{name}_var", values=np.ones_like(scale_values)),
        ],
        outputs=[gs.Variable(name=f"{name}_output")],
        name=name,
    )


def _next_node(node: gs.Node, op: str) -> gs.Node | None:
    """Return the first ``op`` node that reads the first output of ``node``."""
    return next((n for n in node.outputs[0].outputs if n.op == op), None)


def _has_padding(conv_node: gs.Node) -> bool:
    """Tell whether a Conv pads its input."""
    pads = cast("list[int]", conv_node.attrs.get("pads", []))
    return any(pads) or conv_node.attrs.get("auto_pad") in [
        "SAME_UPPER",
        "SAME_LOWER",
    ]


def _conv_after(node: gs.Node, op: str) -> tuple[gs.Node, gs.Node] | None:
    """Find the chain ``node -> op -> Conv``, with a Conv that does not pad.

    Returns:
        The ``op`` node and the Conv, or ``None``.
    """
    second = _next_node(node, op)
    if second is None:
        return None
    conv_node = _next_node(second, "Conv")
    if conv_node is None or _has_padding(conv_node):
        return None
    return second, conv_node


def _scale_conv_weights(conv_node: gs.Node, mul_value: np.ndarray) -> None:
    """Fold the value of a ``Mul`` before a Conv into the Conv weights."""
    conv_weights = conv_node.inputs[1]
    conv_node.inputs[1] = gs.Constant(
        name=conv_weights.name, values=conv_weights.values * mul_value
    )


def _shift_conv_bias(
    conv_node: gs.Node, add_value: np.ndarray, weights: gs.Constant
) -> None:
    """Fold the value of an ``Add`` before a Conv into the Conv bias.

    ``weights`` are the Conv weights that the added value passes
    through.
    """
    shift = np.sum(add_value * weights.values, axis=(1, 2, 3))
    if len(conv_node.inputs) > 2:
        conv_bias = conv_node.inputs[2]
        new_bias = conv_bias.values + shift
        if new_bias.shape != conv_bias.values.shape:
            raise ValueError(
                f"New bias shape: {new_bias.shape} != Old bias shape: {conv_bias.values.shape}"
            )
        conv_node.inputs[2] = gs.Constant(name=conv_bias.name, values=new_bias)
        return

    if shift.shape != weights.shape[0]:
        raise ValueError(
            f"New bias shape: {shift.shape} != Conv weights shape: {weights.shape[0]}"
        )
    conv_node.inputs.append(
        gs.Constant(name=f"{conv_node.name}_bias", values=shift)
    )


def _split_concat_conv(
    split_node: gs.Node,
) -> tuple[gs.Node, list[gs.Node]] | None:
    """Find the chain from a ``Split`` through a ``Concat`` to a Conv.

    After the ``Concat``, the chain follows the first reader of each
    first output.

    Returns:
        The ``Concat`` node and the nodes after it up to the Conv, or
        ``None`` when the chain ends before a Conv.
    """
    concat_node = _next_node(split_node, "Concat")
    if concat_node is None:
        return None
    path = []
    current_node = concat_node
    while current_node.op != "Conv":
        next_node = next(iter(current_node.outputs[0].outputs), None)
        if next_node is None:
            return None
        current_node = next_node
        path.append(current_node)
    return concat_node, path


def _output_names(graph: gs.Graph) -> list[str]:
    """Return the names of the outputs of the graph.

    GraphSurgeon declares an output as the abstract ``gs.Tensor``, which
    has no ``name``. Only the two concrete subclasses carry one, and an
    imported graph holds nothing else.

    Args:
        graph: Graph whose outputs are named.

    Returns:
        The output names, in the order the graph lists them.

    """
    names = []
    for tensor in graph.outputs:
        assert isinstance(tensor, gs.Variable | gs.Constant)
        names.append(tensor.name)
    return names
