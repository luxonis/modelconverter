import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from loguru import logger
from onnx import TensorProto, checker, helper
from onnxsim import simplify

from modelconverter.utils.config import InputConfig
from modelconverter.utils.onnx_compatibility import (
    ensure_onnx_helper_compatibility,
    has_external_data,
    save_onnx_model,
)

from .exceptions import ONNXException

ensure_onnx_helper_compatibility()

# GraphSurgeon still imports helper conversion functions removed in ONNX 1.21.
# Patch them back in before importing GraphSurgeon.
import onnx_graphsurgeon as gs  # noqa: E402


def get_opset_version(model: onnx.ModelProto) -> int:
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
    if not any(
        cfg.requires_onnx_input_modification(reverse_only=reverse_only)
        for cfg in input_configs.values()
    ):
        logger.info(
            "No ONNX input normalization changes requested; using original model."
        )
        return model_path

    model = onnx.load(str(model_path))
    model_has_external_data = has_external_data(model_path)

    graph = model.graph

    new_nodes = []
    new_initializers = []
    input_names = [input_tensor.name for input_tensor in graph.input]
    if not all(name in input_names for name in input_configs):
        raise ONNXException(
            "You either used an invalid input name, or you're attempting "
            "to use a hidden network node as an input. This is not supported "
            "in combination with input modifications (mean, scale, etc.). "
            "Either use an actual input name, or modify your network."
        )

    for input_tensor in graph.input:
        input_name = input_tensor.name
        input_dtype = input_tensor.type.tensor_type.elem_type
        if input_name not in input_configs:
            continue
        cfg = input_configs[input_name]
        if not cfg.requires_onnx_input_modification(reverse_only=reverse_only):
            continue

        shape = cfg.shape
        layout = cfg.layout or "NCHW"
        if layout not in ["NCHW", "NHWC"]:
            logger.warning(
                f"Input '{input_name}' has layout '{layout}', "
                "but only 'NCHW' and 'NHWC' are supported for normalization. "
                "Skipping."
            )
            continue

        if shape is not None:
            n_channels = shape[layout.index("C")]
            if n_channels != 3:
                logger.warning(
                    f"Input '{input_name}' has {n_channels} channels, "
                    "but normalization is only supported for 3 channels. "
                    "Skipping."
                )
                continue

        last_output = input_name

        # 1. Reverse channels if needed
        if cfg.encoding_mismatch:
            opset = get_opset_version(model)
            split_names = [f"split_{i}_{input_name}" for i in range(3)]
            axis = 1 if layout == "NCHW" else 3

            if opset < 13:
                split_node = helper.make_node(
                    "Split",
                    inputs=[last_output],
                    outputs=split_names,
                    axis=axis,
                    split=[1] * n_channels,
                    name=f"split_{input_name}",
                )
            else:
                split_lengths_name = f"split_{input_name}_lengths"
                split_lengths_tensor = helper.make_tensor(
                    name=split_lengths_name,
                    data_type=TensorProto.INT64,
                    dims=[n_channels],
                    vals=[1] * n_channels,
                )
                new_initializers.append(split_lengths_tensor)
                split_node = helper.make_node(
                    "Split",
                    inputs=[last_output, split_lengths_name],
                    outputs=split_names,
                    axis=axis,
                    name=f"split_{input_name}",
                )
            new_nodes.append(split_node)

            concat_node = helper.make_node(
                "Concat",
                inputs=split_names[::-1],
                outputs=[f"normalized_{input_name}"],
                axis=axis,
                name=f"concat_{input_name}",
            )
            new_nodes.append(concat_node)
            last_output = f"normalized_{input_name}"

        # 2. Subtract (mean) if mean_values is not None and not all 0
        if (
            not reverse_only
            and cfg.mean_values is not None
            and any(v != 0 for v in cfg.mean_values)
        ):
            if cfg.encoding_mismatch:
                cfg.mean_values = cfg.mean_values[::-1]

            sub_out = f"sub_out_{input_name}"
            sub_node = helper.make_node(
                "Sub",
                inputs=[last_output, f"mean_{input_name}"],
                outputs=[sub_out],
                name=f"sub_out_{input_name}",
            )
            new_nodes.append(sub_node)
            last_output = sub_out

            mean_tensor = helper.make_tensor(
                f"mean_{input_name}",
                input_dtype,
                [1, len(cfg.mean_values), 1, 1]
                if layout == "NCHW"
                else [1, 1, 1, len(cfg.mean_values)],
                cfg.mean_values,
            )
            new_initializers.append(mean_tensor)

        # 3. Divide (scale) if scale_values is not None and not all 1
        if (
            not reverse_only
            and cfg.scale_values is not None
            and any(v != 1 for v in cfg.scale_values)
        ):
            if cfg.encoding_mismatch:
                cfg.scale_values = cfg.scale_values[::-1]

            div_out = f"div_out_{input_name}"
            div_node = helper.make_node(
                "Mul",
                inputs=[last_output, f"scale_{input_name}"],
                outputs=[div_out],
                name=f"div_out_{input_name}",
            )
            new_nodes.append(div_node)
            last_output = div_out

            scale_tensor = helper.make_tensor(
                f"scale_{input_name}",
                input_dtype,
                [1, len(cfg.scale_values), 1, 1]
                if layout == "NCHW"
                else [1, 1, 1, len(cfg.scale_values)],
                [1 / v for v in cfg.scale_values],
            )
            new_initializers.append(scale_tensor)

        # Update input of other nodes to use the last output
        for node in graph.node:
            new_inputs = [
                last_output if inp == input_name else inp for inp in node.input
            ]
            del node.input[:]
            node.input.extend(new_inputs)

        # Insert the new nodes into the graph at the appropriate position
        idx = next(
            (
                i
                for i, node in enumerate(graph.node)
                if last_output in node.input
            ),
            0,
        )

        nodes_as_list = list(graph.node)
        nodes_as_list[idx:idx] = new_nodes
        del graph.node[:]
        graph.node.extend(nodes_as_list)

        new_nodes.clear()

    graph.initializer.extend(new_initializers)

    save_onnx_model(
        model,
        save_path,
        save_as_external_data=model_has_external_data,
        location=f"{save_path.name}_data",
    )

    checker.check_model(str(save_path))

    return save_path


class ONNXModifier:
    """ONNX model modifier class to optimize and modify the ONNX model.

    Attributes:
    ----------
    model_path : Path
        Path to the base ONNX model
    output_path : Path
        Path to save the modified ONNX model
    """

    def __init__(
        self,
        model_path: Path,
        output_path: Path,
        skip_optimization: bool = False,
        skip_constant_folding: bool = False,
    ) -> None:
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

        @param substitute_sub_with_add: Replace Sub nodes with Add
            nodes.
        @type substitute_sub_with_add: bool
        @param substitute_div_with_mul: Replace Div nodes with Mul
            nodes.
        @type substitute_div_with_mul: bool
        @param fuse_add_mul_to_bn: Fuse Add and Mul nodes into
            BatchNormalization nodes.
        @type fuse_add_mul_to_bn: bool
        @param fuse_comb_add_mul_to_conv: Fuse a combined Add and Mul
            pair into a Conv node.
        @type fuse_comb_add_mul_to_conv: bool
        @param fuse_single_add_mul_to_conv: Fuse a single Add or Mul
            node into a Conv node.
        @type fuse_single_add_mul_to_conv: bool
        @param fuse_split_concat_to_conv: Fuse Split and Concat nodes
            into Conv nodes.
        @type fuse_split_concat_to_conv: bool
        @rtype: bool
        @return: C{True} if the model was modified and exported.
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

        @param from_modelproto: Compare the model held in memory against
            the previous one. If C{False}, compare the file at
            C{model_path} against the file at C{output_path}.
        @type from_modelproto: bool
        @rtype: bool
        @return: C{True} if every output of the two models agrees.
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

        inputs = {}
        for input in ort_session_1.get_inputs():
            if input.type == "tensor(float64)":
                input_type = np.float64
            elif input.type in {"tensor(float32)", "tensor(float)"}:
                input_type = np.float32
            elif input.type == "tensor(float16)":
                input_type = np.float16
            elif input.type == "tensor(int64)":
                input_type = np.int64
            elif input.type == "tensor(int32)":
                input_type = np.int32
            elif input.type == "tensor(int16)":
                input_type = np.int16
            elif input.type == "tensor(int8)":
                input_type = np.int8
            elif input.type == "tensor(bool)":
                input_type = "bool"

            inputs[input.name] = np.random.rand(*input.shape).astype(
                input_type
            )

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
        """Load the ONNX model and store it as onnx.ModelProto and
        onnx_graphsurgeon.GraphSurgeon graph."""
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
        """Add output nodes to the ONNX model.

        @param output_names: List of output node names to add to the
            ONNX model
        @type output_names: List[str]
        """
        graph_outputs = _output_names(self._onnx_gs)
        for name, tensor in self._onnx_gs.tensors().items():
            if name in output_names and name not in graph_outputs:
                self._onnx_gs.outputs.append(tensor)
        self._onnx_model = gs.export_onnx(self._onnx_gs)

    def _get_constant_map(self, graph: gs.Graph) -> dict[str, np.ndarray]:
        """Extract constant tensors from the GraphSurgeon graph.

        @param graph: GraphSurgeon graph
        @type graph: gs.Graph
        @return: Constant tensor map with tensor name as key and tensor
            value as value
        @rtype: Dict[str, np.ndarray]
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
        """Returns the constant value of a node if it is a constant
        node.

        @param node: Node to check
        @type node: gs.Node
        @param constant_map: Constant tensor map with tensor name as key
            and tensor value as value
        @type constant_map: Dict[str, np.ndarray]
        @return: Constant tensor value and index
        @rtype: Optional[Tuple[np.ndarray, int]]
        """
        for idx, input in enumerate(node.inputs):
            if input.name in constant_map:
                return (constant_map[input.name], idx)

        return None

    @staticmethod
    def _get_variable_input(node: gs.Node) -> tuple[gs.Variable, int] | None:
        """Returns the variable input of a node.

        @param node: Node to check
        @type node: gs.Node
        @return: Variable input and index
        @rtype: Optional[Tuple[gs.Variable, int]]
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
        """Cleanup the graph by adding new nodes, removing old nodes,
        and fixing connections.

        @param nodes_to_add: List of nodes to add to the graph
        @type nodes_to_add: List[gs.Node]
        @param nodes_to_remove: List of nodes to remove from the graph
        @type nodes_to_remove: List[gs.Node]
        @param connections_to_fix: List of connections to fix in the
            graph
        @type connections_to_fix: List[Tuple[gs.Variable, gs.Variable]]
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
        node of a different type. Currently, only Sub -> Add and Div ->
        Mul substitutions are allowed.

        @param source_node: Source node type to substitute
        @type source_node: str
        @param target_node: Target node type to substitute with
        @type target_node: str
        """
        if source_node not in ["Sub", "Div"] or target_node not in [
            "Add",
            "Mul",
        ]:
            raise ValueError(
                "Invalid source or target node type. Valid source types: Sub, Div. Valid target types: Add, Mul."
            )

        if (source_node == "Sub" and target_node == "Mul") or (
            source_node == "Div" and target_node == "Add"
        ):
            raise ValueError(
                "Invalid substitution. Available substitutions: Sub -> Add, Div -> Mul"
            )

        constant_map = self._get_constant_map(self._onnx_gs)

        def create_new_node(
            node: gs.Node, target_node: str, const_idx: int
        ) -> gs.Node | None:
            if const_idx == 0:
                return None

            first_input = node.inputs[0]
            second_input = node.inputs[const_idx]
            if target_node == "Add":
                new_cost_val = -second_input.values
                return gs.Node(
                    op="Add",
                    inputs=[
                        first_input,
                        gs.Constant(
                            name=f"{node.name}_{second_input.name}/Substitute",
                            values=np.array(
                                new_cost_val, dtype=second_input.dtype
                            ),
                        ),
                    ],
                    outputs=[gs.Variable(name=f"{node.name}/Add_output")],
                    name=f"{node.name}/To_Add",
                )
            if target_node == "Mul":
                new_cost_val = 1.0 / second_input.values
                if second_input.dtype not in [
                    np.float16,
                    np.float32,
                    np.float64,
                ]:
                    return None
                return gs.Node(
                    op="Mul",
                    inputs=[
                        first_input,
                        gs.Constant(
                            name=f"{node.name}_{second_input.name}/Substitute",
                            values=np.array(
                                new_cost_val, dtype=second_input.dtype
                            ),
                        ),
                    ],
                    outputs=[gs.Variable(name=f"{node.name}/Mul_output")],
                    name=f"{node.name}/To_Mul",
                )

        nodes_to_add = []
        nodes_to_remove = []
        connections_to_fix = []

        graph_output_names = set(_output_names(self._onnx_gs))

        for node in self._onnx_gs.nodes:
            if node.op == source_node:
                next_nodes = [
                    n
                    for n in self._onnx_gs.nodes
                    if node.outputs[0] in n.inputs
                ]
                # Skip optimization for nodes followed by Erf operations due to conversion compatibility issues with SNPE 2.32.6.
                if any(n.op == "Erf" for n in next_nodes):
                    logger.warning(
                        f"Skipping the `{source_node} -> {target_node}` substitution "
                        f"optimization for node '{node.name}' with op '{node.op}' "
                        "because it is followed by an 'Erf' node."
                    )
                    continue
                constant = self._get_constant_value(node, constant_map)
                if constant is not None:
                    _, const_idx = constant
                    new_node = create_new_node(node, target_node, const_idx)
                    if new_node is not None:
                        nodes_to_add.append(new_node)
                        connections_to_fix.append(
                            (
                                node.outputs[0],
                                new_node.outputs[0],
                            )
                        )

                        if node.outputs[0].name in graph_output_names:
                            for i, graph_output_name in enumerate(
                                _output_names(self._onnx_gs)
                            ):
                                if graph_output_name == node.outputs[0].name:
                                    new_output_var = gs.Variable(
                                        name=node.outputs[0].name,
                                        dtype=node.outputs[0].dtype,
                                        shape=node.outputs[0].shape,
                                    )
                                    new_node.outputs[0] = new_output_var
                                    self._onnx_gs.outputs[i] = new_output_var
                                    break

                        nodes_to_remove.append(node)

        if not any([nodes_to_add, nodes_to_remove, connections_to_fix]):
            logger.warning(
                "No applicable Sub-Add or Div-Mul pattern found for substitution."
            )
            return

        self._graph_cleanup(nodes_to_add, nodes_to_remove, connections_to_fix)
        self._onnx_model = gs.export_onnx(self._onnx_gs)

        self._optimize_onnx()

    def _fuse_add_mul_to_bn(self) -> None:
        """Fuse Add/Sub and Mul nodes that come immediately after a Conv
        node into a BatchNormalization node.

        The fusion patterns considered are:
        1. Conv -> Add -> Mul
        2. Conv -> Mul -> Add
        3. Conv -> Mul
        4. Conv -> Add
        """
        FUSION_PATTERNS = [
            ("Conv", "Add", "Mul"),
            ("Conv", "Mul", "Add"),
            ("Conv", "Mul"),
            ("Conv", "Add"),
        ]

        constant_map = self._get_constant_map(self._onnx_gs)

        def create_batch_norm_node(
            name: str,
            input_tensor: gs.Variable,
            scale: float | np.ndarray,
            bias: float | np.ndarray,
        ) -> gs.Node:
            assert input_tensor.shape is not None
            conv_channels = int(input_tensor.shape[1])
            scale_values = np.array(
                [scale] * conv_channels, dtype=self._dtype
            ).squeeze()
            bias_values = np.array(
                [bias] * conv_channels, dtype=self._dtype
            ).squeeze()
            mean_values = np.zeros_like(scale_values)
            var_values = np.ones_like(scale_values)
            scale_tensor = gs.Constant(
                name=f"{name}_scale",
                values=scale_values,
            )
            bias_tensor = gs.Constant(
                name=f"{name}_bias",
                values=bias_values,
            )
            mean_tensor = gs.Constant(
                name=f"{name}_mean",
                values=mean_values,
            )
            var_tensor = gs.Constant(
                name=f"{name}_var",
                values=var_values,
            )
            return gs.Node(
                op="BatchNormalization",
                inputs=[
                    input_tensor,
                    scale_tensor,
                    bias_tensor,
                    mean_tensor,
                    var_tensor,
                ],
                outputs=[gs.Variable(name=f"{name}_output")],
                name=name,
            )

        all_sequences = []

        for pattern in FUSION_PATTERNS:
            for node in self._onnx_gs.nodes:
                if node.op != pattern[0]:
                    continue

                sequence = [node]
                current_node = node
                for op_type in pattern[1:]:
                    next_nodes = [
                        n
                        for n in self._onnx_gs.nodes
                        if n.inputs
                        and current_node.outputs[0] in n.inputs
                        and n.op == op_type
                    ]
                    if not next_nodes:
                        break
                    current_node = next_nodes[0]
                    sequence.append(current_node)

                if len(sequence) == len(pattern):
                    all_sequences.append(sequence)

        longest_sequences = []
        for seq in all_sequences:
            is_subset = any(
                all(node in longer_seq for node in seq)
                and len(seq) < len(longer_seq)
                for longer_seq in all_sequences
            )
            if not is_subset:
                longest_sequences.append(seq)

        nodes_to_add = []
        nodes_to_remove = []
        connections_to_fix = []

        for sequence in longest_sequences:
            valid_fusion = True
            scale, bias = 1.0, 0.0

            conv_node = None
            for seq_node in sequence:
                if seq_node.op == "Conv":
                    conv_node = seq_node
                    continue

                constant = self._get_constant_value(seq_node, constant_map)
                if constant is None:
                    valid_fusion = False
                    break

                constant_val, _ = constant

                if seq_node.op == "Add":
                    bias += constant_val
                elif seq_node.op == "Sub":
                    bias -= constant_val
                elif seq_node.op == "Mul":
                    scale *= constant_val

            if (
                not valid_fusion
                or not conv_node
                or len(conv_node.outputs[0].outputs) > 1
            ):
                continue

            bn_name = f"BatchNorm_{conv_node.name.replace('/', '', 1)}"

            bn_node = create_batch_norm_node(
                bn_name, conv_node.outputs[0], scale, bias
            )
            nodes_to_add.append(bn_node)

            if sequence[0].op == "Conv":
                connections_to_fix.append(
                    (
                        sequence[-1].outputs[0],
                        bn_node.outputs[0],
                    )
                )

            nodes_to_remove.extend(
                seq_node for seq_node in sequence if seq_node.op != "Conv"
            )

        if not any([nodes_to_add, nodes_to_remove, connections_to_fix]):
            logger.warning(
                "No applicable Conv-Add-Mul pattern found for batch normalization fusion."
            )
            return

        self._graph_cleanup(nodes_to_add, nodes_to_remove, connections_to_fix)
        self._onnx_model = gs.export_onnx(self._onnx_gs)

        self._optimize_onnx()

    def _fuse_single_add_mul_to_conv(self) -> None:
        """Fuse Add and Mul nodes that precede a Conv node directly into
        the Conv node."""
        nodes_to_remove = []
        connections_to_fix = []

        constant_map = self._get_constant_map(self._onnx_gs)

        for node in self._onnx_gs.nodes:
            if node.op == "Mul":
                mul_node = node
                if len(mul_node.outputs[0].outputs) > 1:
                    continue

                conv_node = next(
                    (n for n in mul_node.outputs[0].outputs if n.op == "Conv"),
                    None,
                )
                if conv_node is None:
                    continue

                constant = self._get_constant_value(mul_node, constant_map)
                if constant is None:
                    continue

                mul_value, _ = constant

                conv_weights = conv_node.inputs[1]

                new_weights = conv_weights.values * mul_value

                conv_node.inputs[1] = gs.Constant(
                    name=conv_weights.name,
                    values=new_weights,
                )

                nodes_to_remove.append(mul_node)

                connections_to_fix.append(
                    (
                        mul_node.outputs[0],
                        mul_node.inputs[0],
                    )
                )

            if node.op == "Add":
                add_node = node
                if len(add_node.outputs[0].outputs) > 1:
                    continue

                conv_node = next(
                    (n for n in add_node.outputs[0].outputs if n.op == "Conv"),
                    None,
                )
                if (
                    conv_node is None
                    or (
                        "pads" in conv_node.attrs
                        and any(conv_node.attrs["pads"])
                    )
                    or (
                        "auto_pad" in conv_node.attrs
                        and conv_node.attrs["auto_pad"]
                        in ["SAME_UPPER", "SAME_LOWER"]
                    )
                ):
                    continue

                constant = self._get_constant_value(add_node, constant_map)
                if constant is None:
                    continue

                add_value, _ = constant

                conv_weights = conv_node.inputs[1]
                conv_bias = (
                    conv_node.inputs[2] if len(conv_node.inputs) > 2 else None
                )

                if conv_bias is not None:
                    new_bias = conv_bias.values + np.sum(
                        add_value * conv_weights.values, axis=(1, 2, 3)
                    )
                    if new_bias.shape != conv_bias.values.shape:
                        raise ValueError(
                            f"New bias shape: {new_bias.shape} != Old bias shape: {conv_bias.values.shape}"
                        )
                else:
                    new_bias = np.sum(
                        add_value * conv_weights.values, axis=(1, 2, 3)
                    )
                    if new_bias.shape != conv_weights.shape[0]:
                        raise ValueError(
                            f"New bias shape: {new_bias.shape} != Conv weights shape: {conv_weights.shape[0]}"
                        )

                if conv_bias is not None:
                    conv_node.inputs[2] = gs.Constant(
                        name=conv_bias.name,
                        values=new_bias,
                    )
                else:
                    conv_node.inputs.append(
                        gs.Constant(
                            name=f"{conv_node.name}_bias",
                            values=new_bias,
                        )
                    )

                nodes_to_remove.append(add_node)

                connections_to_fix.append(
                    (
                        add_node.outputs[0],
                        add_node.inputs[0],
                    )
                )

        if not any([nodes_to_remove, connections_to_fix]):
            logger.warning(
                "No applicable Add-Mul-Conv pattern found for fusion."
            )
            return

        self._graph_cleanup([], nodes_to_remove, connections_to_fix)
        self._onnx_model = gs.export_onnx(self._onnx_gs)

        self._optimize_onnx()

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
                mul_node = node

                add_node = next(
                    (n for n in mul_node.outputs[0].outputs if n.op == "Add"),
                    None,
                )
                if add_node is None:
                    continue

                conv_node = next(
                    (n for n in add_node.outputs[0].outputs if n.op == "Conv"),
                    None,
                )
                if (
                    conv_node is None
                    or (
                        "pads" in conv_node.attrs
                        and any(conv_node.attrs["pads"])
                    )
                    or (
                        "auto_pad" in conv_node.attrs
                        and conv_node.attrs["auto_pad"]
                        in ["SAME_UPPER", "SAME_LOWER"]
                    )
                ):
                    continue

                constant = self._get_constant_value(mul_node, constant_map)
                if constant is None:
                    continue
                mul_value, _ = constant

                constant = self._get_constant_value(add_node, constant_map)
                if constant is None:
                    continue
                add_value, _ = constant

                conv_weights = conv_node.inputs[1]
                conv_bias = (
                    conv_node.inputs[2] if len(conv_node.inputs) > 2 else None
                )

                new_weights = conv_weights.values * mul_value

                conv_node.inputs[1] = gs.Constant(
                    name=conv_weights.name,
                    values=new_weights,
                )

                if conv_bias is not None:
                    new_bias = conv_bias.values + np.sum(
                        add_value * conv_weights.values, axis=(1, 2, 3)
                    )
                    if new_bias.shape != conv_bias.values.shape:
                        raise ValueError(
                            f"New bias shape: {new_bias.shape} != Old bias shape: {conv_bias.values.shape}"
                        )
                    conv_node.inputs[2] = gs.Constant(
                        name=conv_bias.name,
                        values=new_bias,
                    )
                else:
                    new_bias = np.sum(
                        add_value * conv_weights.values, axis=(1, 2, 3)
                    )
                    if new_bias.shape != conv_weights.shape[0]:
                        raise ValueError(
                            f"New bias shape: {new_bias.shape} != Conv weights shape: {conv_weights.shape[0]}"
                        )
                    conv_node.inputs.append(
                        gs.Constant(
                            name=f"{conv_node.name}_bias",
                            values=new_bias,
                        )
                    )

                variable = self._get_variable_input(mul_node)
                if variable is None:
                    continue
                _, mul_idx = variable

                nodes_to_remove.append(mul_node)
                nodes_to_remove.append(add_node)

                connections_to_fix.append(
                    (
                        add_node.outputs[0],
                        mul_node.inputs[mul_idx],
                    )
                )

            if node.op == "Add":
                add_node = node

                mul_node = next(
                    (n for n in add_node.outputs[0].outputs if n.op == "Mul"),
                    None,
                )
                if mul_node is None:
                    continue

                conv_node = next(
                    (n for n in mul_node.outputs[0].outputs if n.op == "Conv"),
                    None,
                )
                if (
                    conv_node is None
                    or (
                        "pads" in conv_node.attrs
                        and any(conv_node.attrs["pads"])
                    )
                    or (
                        "auto_pad" in conv_node.attrs
                        and conv_node.attrs["auto_pad"]
                        in ["SAME_UPPER", "SAME_LOWER"]
                    )
                ):
                    continue

                constant = self._get_constant_value(add_node, constant_map)
                if constant is None:
                    continue
                add_value, _ = constant

                constant = self._get_constant_value(mul_node, constant_map)
                if constant is None:
                    continue
                mul_value, _ = constant

                add_value *= mul_value

                conv_weights = conv_node.inputs[1]
                conv_bias = (
                    conv_node.inputs[2] if len(conv_node.inputs) > 2 else None
                )

                if conv_bias is not None:
                    new_bias = conv_bias.values + np.sum(
                        add_value * conv_weights.values, axis=(1, 2, 3)
                    )
                    if new_bias.shape != conv_bias.values.shape:
                        raise ValueError(
                            f"New bias shape: {new_bias.shape} != Old bias shape: {conv_bias.values.shape}"
                        )
                    conv_node.inputs[2] = gs.Constant(
                        name=conv_bias.name,
                        values=new_bias,
                    )
                else:
                    new_bias = np.sum(
                        add_value * conv_weights.values, axis=(1, 2, 3)
                    )
                    if new_bias.shape != conv_weights.shape[0]:
                        raise ValueError(
                            f"New bias shape: {new_bias.shape} != Conv weights shape: {conv_weights.shape[0]}"
                        )
                    conv_node.inputs.append(
                        gs.Constant(
                            name=f"{conv_node.name}_bias",
                            values=new_bias,
                        )
                    )

                new_weights = conv_weights.values * mul_value

                conv_node.inputs[1] = gs.Constant(
                    name=conv_weights.name,
                    values=new_weights,
                )

                variable = self._get_variable_input(add_node)
                if variable is None:
                    continue
                _, add_idx = variable

                nodes_to_remove.append(add_node)
                nodes_to_remove.append(mul_node)

                connections_to_fix.append(
                    (
                        mul_node.outputs[0],
                        add_node.inputs[add_idx],
                    )
                )

        if not any([nodes_to_remove, connections_to_fix]):
            logger.warning(
                "No applicable Add-Mul-Conv pattern found for fusion."
            )
            return
        self._graph_cleanup([], nodes_to_remove, connections_to_fix)
        self._onnx_model = gs.export_onnx(self._onnx_gs)

        self._optimize_onnx()

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

            if node.op == "Split":
                split_node = node

                concat_node = next(
                    (
                        n
                        for n in split_node.outputs[0].outputs
                        if n.op == "Concat"
                    ),
                    None,
                )
                if concat_node is None:
                    continue

                intermediate_nodes = []
                current_node = concat_node
                while current_node.op != "Conv":
                    next_node = next(
                        iter(current_node.outputs[0].outputs), None
                    )
                    if next_node is None:
                        break
                    current_node = next_node
                    intermediate_nodes.append(current_node)

                if not intermediate_nodes:
                    continue

                conv_node = intermediate_nodes[-1]
                if conv_node.op != "Conv":
                    continue

                conv_weights = conv_node.inputs[1]

                if split_node.attrs["axis"] != concat_node.attrs["axis"]:
                    raise ValueError(
                        f"Split and Concat axis mismatch: {split_node.attrs['axis']} != {concat_node.attrs['axis']}"
                    )

                channels_axis = split_node.attrs["axis"]
                if not isinstance(channels_axis, int):
                    raise ValueError(
                        f"Split node axis must be an integer, got: {channels_axis}"
                    )
                if conv_weights.shape[channels_axis] not in [1, 3]:
                    break

                for inter_node in intermediate_nodes[:-1]:
                    constant = self._get_constant_value(
                        inter_node, self._get_constant_map(self._onnx_gs)
                    )
                    if constant is None:
                        continue
                    constant_value, constant_idx = constant
                    if constant_value.ndim == 1:
                        continue

                    if (
                        constant_value.shape[channels_axis]
                        != conv_weights.values.shape[1]
                    ):
                        logger.warning(
                            f"Spatial dimensions mismatch between Conv and intermediate node {inter_node.name}: {constant_value.shape[channels_axis]} != {conv_weights.values.shape[1]}, discarding this step."
                        )

                    inter_node.inputs[constant_idx].values = np.flip(
                        constant_value, axis=channels_axis
                    )

                conv_weights.values = np.flip(
                    conv_weights.values, axis=channels_axis
                )

                nodes_to_remove.append(split_node)
                nodes_to_remove.append(concat_node)

                connections_to_fix.append(
                    (
                        concat_node.outputs[0],
                        split_node.inputs[0],
                    )
                )

                break

        if not any([nodes_to_remove, connections_to_fix]):
            logger.warning(
                "No applicable Split-Conv-Concat pattern found for fusion."
            )
            return

        self._graph_cleanup([], nodes_to_remove, connections_to_fix)
        self._onnx_model = gs.export_onnx(self._onnx_gs)

        self._optimize_onnx()

    def _revert_changes(self) -> None:
        """Reverts ONNX model to previous state."""
        self._onnx_model = self._prev_onnx_model
        self._onnx_gs = self._prev_onnx_gs

    def _apply_optimization_step(
        self, step_name: str, optimization_func: Callable
    ) -> None:
        """Applies a single optimization step to the ONNX model.

        @param step_name: Name of the optimization step
        @type step_name: str
        @param optimization_func: Optimization function to apply
        @type optimization_func: Callable
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


def _output_names(graph: gs.Graph) -> list[str]:
    """The names of the outputs of the graph.

    GraphSurgeon declares an output as the abstract C{gs.Tensor}, which
    has no C{name}. Only the two concrete subclasses carry one, and an
    imported graph holds nothing else.
    """
    names = []
    for tensor in graph.outputs:
        assert isinstance(tensor, gs.Variable | gs.Constant)
        names.append(tensor.name)
    return names
