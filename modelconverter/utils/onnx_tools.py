import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxoptimizer
from onnx import checker, helper
from onnx.onnx_pb import TensorProto
from onnxsim import simplify

from modelconverter.utils.config import InputConfig

from .exceptions import ONNXException

logger = logging.getLogger(__name__)


def onnx_attach_normalization_to_inputs(
    model_path: Path,
    save_path: Path,
    input_configs: Dict[str, InputConfig],
    *,
    reverse_only=False,
) -> Path:
    model = onnx.load(str(model_path))

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
        if input_name not in input_configs:
            continue
        cfg = input_configs[input_name]
        if (
            cfg.encoding.from_ == cfg.encoding.to
            and cfg.mean_values is None
            and cfg.scale_values is None
        ):
            continue

        shape = cfg.shape
        layout = cfg.layout or "NCHW"
        if shape is not None:
            n_channels = shape[layout.index("C")]
            if n_channels != 3:
                logger.warning(
                    f"Input '{input_name}' has {n_channels} channels, "
                    "but normalization is only supported for 3 channels. "
                    "Skipping."
                )
                continue

        if layout not in ["NCHW", "NHWC"]:
            logger.warning(
                f"Input '{input_name}' has layout '{layout}', "
                "but only 'NCHW' and 'NHWC' are supported for normalization. "
                "Skipping."
            )
            continue

        last_output = input_name

        # 1. Reverse channels if needed
        if cfg.encoding_mismatch:
            split_names = [f"split_{i}_{input_name}" for i in range(3)]
            split_node = helper.make_node(
                "Split",
                inputs=[last_output],
                outputs=split_names,
                axis=1 if layout == "NCHW" else 3,
            )
            new_nodes.append(split_node)

            concat_node = helper.make_node(
                "Concat",
                inputs=split_names[::-1],
                outputs=[f"normalized_{input_name}"],
                axis=1 if layout == "NCHW" else 3,
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
            )
            new_nodes.append(sub_node)
            last_output = sub_out

            mean_tensor = helper.make_tensor(
                f"mean_{input_name}",
                TensorProto.FLOAT,
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
            )
            new_nodes.append(div_node)
            last_output = div_out

            scale_tensor = helper.make_tensor(
                f"scale_{input_name}",
                TensorProto.FLOAT,
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

    checker.check_model(model)

    onnx.save(model, str(save_path))
    return save_path


@dataclass
class ONNXModifier:
    """ONNX model modifier class to optimize and modify the ONNX model.

    Attributes:
    ----------
    model_path : Path
        Path to the base ONNX model
    output_path : Path
        Path to save the modified ONNX model
    """

    model_path: Path
    output_path: Path

    def __post_init__(self):
        self.load_onnx()

    def load_onnx(self) -> None:
        """Load the ONNX model and store it as onnx.ModelProto and
        onnx_graphsurgeon.GraphSurgeon graph."""

        logger.info(f"Loading model: {self.model_path.stem}")

        # Load the ONNX model
        self.onnx_model, _ = simplify(
            self.model_path.as_posix(), perform_optimization=True
        )

        # Load the ONNX model as a GraphSurgeon graph
        self.onnx_gs = gs.import_onnx(self.onnx_model)

    def optimize_onnx(self, passes: Optional[List[str]] = None) -> None:
        """Optimize and simplify the ONNX model's graph.

        @param passes: List of optimization passes to apply to the ONNX model
        @type passes: Optional[List[str]]
        """

        # Optimize the onnx model based on the optimization passes
        optimised_onnx_model = onnxoptimizer.optimize(
            self.onnx_model, passes=passes
        )

        # Simplify the onnx model
        optimised_onnx_model, _ = simplify(
            optimised_onnx_model, perform_optimization=False
        )

        # Check the model
        onnx.checker.check_model(optimised_onnx_model)

        self.onnx_model, self.onnx_gs = (
            optimised_onnx_model,
            gs.import_onnx(optimised_onnx_model),
        )

    def export_onnx(self, passes: Optional[List[str]] = None) -> None:
        """Export the modified ONNX model to the output path.

        @param passes: List of optimization passes to apply to the ONNX model
        @type passes: Optional[List[str]]
        """

        self.optimize_onnx(passes)

        # Save the modified model
        onnx.save(self.onnx_model, self.output_path)

    def add_outputs(self, output_names: List[str]) -> None:
        """Add output nodes to the ONNX model.

        @param output_names: List of output node names to add to the ONNX model
        @type output_names: List[str]
        """

        # Add output nodes to the graph
        graph_outputs = [output.name for output in self.onnx_gs.outputs]
        for name, tensor in self.onnx_gs.tensors().items():
            if name in output_names and name not in graph_outputs:
                self.onnx_gs.outputs.append(tensor)
        self.onnx_model = gs.export_onnx(self.onnx_gs)

    def get_constant_map(self, graph: gs.Graph) -> Dict[str, np.ndarray]:
        """Extract constant tensors from the GraphSurgeon graph.

        @param graph: GraphSurgeon graph
        @type graph: gs.Graph
        @return: Constant tensor map with tensor name as key and tensor value as value
        @rtype: Dict[str, np.ndarray]
        """

        return {
            tensor.name: tensor.values
            for tensor in graph.tensors().values()
            if isinstance(tensor, gs.Constant)
        }

    @staticmethod
    def get_constant_value(
        node: gs.Node, constant_map: Dict[str, np.ndarray]
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Returns the constant value of a node if it is a constant node.

        @param node: Node to check
        @type node: gs.Node
        @param constant_map: Constant tensor map with tensor name as key and tensor
            value as value
        @type constant_map: Dict[str, np.ndarray]
        @return: Constant tensor value and index
        @rtype: Optional[Tuple[np.ndarray, int]]
        """

        for idx, input in enumerate(node.inputs):
            if input.name in constant_map:
                return (constant_map[input.name], idx)

        return None

    @staticmethod
    def get_variable_input(node: gs.Node) -> Optional[Tuple[gs.Variable, int]]:
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

    def graph_cleanup(
        self,
        nodes_to_add: List[gs.Node],
        nodes_to_remove: List[gs.Node],
        connections_to_fix: List[Tuple[gs.Variable, gs.Variable]],
    ) -> None:
        """Cleanup the graph by adding new nodes, removing old nodes, and fixing
        connections.

        @param nodes_to_add: List of nodes to add to the graph
        @type nodes_to_add: List[gs.Node]
        @param nodes_to_remove: List of nodes to remove from the graph
        @type nodes_to_remove: List[gs.Node]
        @param connections_to_fix: List of connections to fix in the graph
        @type connections_to_fix: List[Tuple[gs.Variable, gs.Variable]]
        """

        # Add new nodes
        for node in nodes_to_add:
            self.onnx_gs.nodes.append(node)

        # Fix input connections
        for old_input, new_input in connections_to_fix:
            for node in self.onnx_gs.nodes:
                for idx, input in enumerate(node.inputs):
                    if input == old_input:
                        node.inputs[idx] = new_input

        # Remove old nodes
        for node in nodes_to_remove:
            self.onnx_gs.nodes.remove(node)

        # Cleanup the graph
        self.onnx_gs.cleanup(
            remove_unused_node_outputs=True, remove_unused_graph_inputs=True
        ).toposort()

    def substitute_node_by_type(
        self, source_node: str, target_node: str
    ) -> None:
        """Substitute a source node of a particular type with a target node of a
        different type. Currently, only Sub -> Add and Div -> Mul substitutions are
        allowed.

        @param source_node: Source node type to substitute
        @type source_node: str
        @param target_node: Target node type to substitute with
        @type target_node: str
        """

        # Allow only substitution from Sub to Add and Div to Mul
        if source_node not in ["Sub", "Div"] or target_node not in [
            "Add",
            "Mul",
        ]:
            raise ValueError(
                "Invalid source or target node type. Valid source types: Sub, Div. Valid target types: Add, Mul."
            )

        # Ensure the combination is valid
        if (
            source_node == "Sub"
            and target_node == "Mul"
            or source_node == "Div"
            and target_node == "Add"
        ):
            raise ValueError(
                "Invalid substitution. Available substitutions: Sub -> Add, Div -> Mul"
            )

        # Extract constant tensors from the graph
        constant_map = self.get_constant_map(self.onnx_gs)

        def create_new_node(
            node: gs.Node, target_node: str, const_idx: int
        ) -> Optional[List[gs.Node]]:
            if const_idx == 0:
                return None
            new_nodes = []
            first_input = node.inputs[0]
            second_input = node.inputs[const_idx]
            if target_node == "Add":
                new_cost_val = -second_input.values
                new_nodes.append(
                    gs.Node(
                        op="Add",
                        inputs=[
                            first_input,
                            gs.Constant(
                                name=f"{second_input.name}/Subtitute",
                                values=np.array(
                                    new_cost_val, dtype=second_input.dtype
                                ),
                            ),
                        ],
                        outputs=[gs.Variable(name=f"{node.name}/Add_output")],
                        name=f"{node.name}/To_Add",
                    )
                )
            elif target_node == "Mul":
                new_cost_val = 1.0 / second_input.values
                if first_input.dtype in [np.float32, np.float64]:
                    new_dtype = first_input.dtype
                else:
                    # If the first input is not a float, cast it to float32 due to the constant value being a float
                    new_dtype = np.float32
                    cast_node = gs.Node(
                        op="Cast",
                        attrs={"to": new_dtype},
                        inputs=[first_input],
                        outputs=[gs.Variable(name=f"{node.name}/Cast_output")],
                        name=f"{node.name}/Cast",
                    )
                    new_nodes.append(cast_node)
                    first_input = cast_node.outputs[0]

                new_nodes.append(
                    gs.Node(
                        op="Mul",
                        inputs=[
                            first_input,
                            gs.Constant(
                                name=f"{second_input.name}/Subtitute",
                                values=np.array(new_cost_val, dtype=new_dtype),
                            ),
                        ],
                        outputs=[gs.Variable(name=f"{node.name}/Mul_output")],
                        name=f"{node.name}/To_Mul",
                    )
                )

            return new_nodes

        nodes_to_add = []
        nodes_to_remove = []
        connections_to_fix = []

        for node in self.onnx_gs.nodes:
            if node.op == source_node:
                constant = self.get_constant_value(node, constant_map)
                if constant is not None:
                    _, const_idx = constant
                    new_nodes = create_new_node(node, target_node, const_idx)
                    if new_nodes is not None:
                        if len(new_nodes) > 1:
                            nodes_to_add.extend(new_nodes)
                            connections_to_fix.append(
                                (
                                    node.outputs[0],
                                    new_nodes[1].outputs[0],
                                )
                            )
                        else:
                            nodes_to_add.append(new_nodes[0])
                            connections_to_fix.append(
                                (
                                    node.outputs[0],
                                    new_nodes[0].outputs[0],
                                )
                            )
                        nodes_to_remove.append(node)

        self.graph_cleanup(nodes_to_add, nodes_to_remove, connections_to_fix)
        self.onnx_model = gs.export_onnx(self.onnx_gs)

        self.optimize_onnx(passes=["fuse_add_bias_into_conv"])

    def fuse_add_mul_to_bn(self) -> None:
        """Fuse Add/Sub and Mul nodes that come immediately after a Conv node into a
        BatchNormalization node.

        The fusion patterns considered are:
        1. Conv -> Add -> Mul
        2. Conv -> Sub -> Mul
        3. Conv -> Mul -> Add
        4. Conv -> Mul -> Sub
        5. Conv -> Mul
        6. Conv -> Add
        7. Conv -> Sub
        """

        FUSION_PATTERNS = [
            ("Conv", "Add", "Mul"),
            ("Conv", "Sub", "Mul"),
            ("Conv", "Mul", "Add"),
            ("Conv", "Mul", "Sub"),
            ("Conv", "Mul"),
            ("Conv", "Add"),
            ("Conv", "Sub"),
        ]

        # Extract constant tensors from the graph
        constant_map = self.get_constant_map(self.onnx_gs)

        def create_batch_norm_node(
            name: str, input_tensor: gs.Variable, scale: float, bias: float
        ) -> gs.Node:
            conv_channels = input_tensor.shape[1]
            scale_values = np.array(
                [scale] * conv_channels, dtype=np.float32
            ).squeeze()
            bias_values = np.array(
                [bias] * conv_channels, dtype=np.float32
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
            bn_node = gs.Node(
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
            return bn_node

        all_sequences = []

        for pattern in FUSION_PATTERNS:
            for node in self.onnx_gs.nodes:
                if node.op != pattern[0]:
                    continue
                # Attempt to match the rest of the pattern
                sequence = [node]
                current_node = node
                for op_type in pattern[1:]:
                    next_nodes = [
                        n
                        for n in self.onnx_gs.nodes
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

        # Filter out sequences that are subsets of others
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
            # Validate if the pattern is feasible for fusion
            valid_fusion = True
            scale, bias = 1.0, 0.0

            conv_node = None
            for seq_node in sequence:
                if seq_node.op == "Conv":
                    conv_node = seq_node
                    continue

                constant = self.get_constant_value(seq_node, constant_map)
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

            for seq_node in sequence:
                if seq_node.op != "Conv":
                    nodes_to_remove.append(seq_node)

        self.graph_cleanup(nodes_to_add, nodes_to_remove, connections_to_fix)
        self.onnx_model = gs.export_onnx(self.onnx_gs)

        self.optimize_onnx(passes=["fuse_bn_into_conv"])

    def fuse_single_add_mul_to_conv(self) -> None:
        """Fuse Add and Mul nodes that precede a Conv node directly into the Conv
        node."""

        nodes_to_remove = []
        connections_to_fix = []

        # Extract constant tensors from the graph
        constant_map = self.get_constant_map(self.onnx_gs)

        # Iterate over the nodes in the graph
        for node in self.onnx_gs.nodes:
            # Look for Mul -> Conv pattern
            if node.op == "Mul":
                mul_node = node
                if len(mul_node.outputs[0].outputs) > 1:
                    continue

                # Check for Conv node
                conv_node = next(
                    (n for n in mul_node.outputs[0].outputs if n.op == "Conv"),
                    None,
                )
                if conv_node is None:
                    continue

                # Get the values of the Add and Mul nodes
                constant = self.get_constant_value(mul_node, constant_map)
                if constant is None:
                    continue

                mul_value, _ = constant

                # Get the weights and bias of the Conv node
                conv_weights = conv_node.inputs[1]

                # Adjust the Conv weights by the Mul value on a per-channel basis on axis 1
                new_weights = conv_weights.values * mul_value

                conv_node.inputs[1] = gs.Constant(
                    name=conv_weights.name,
                    values=new_weights,
                )

                # Remove Mul nodes
                nodes_to_remove.append(mul_node)

                connections_to_fix.append(
                    (mul_node.outputs[0], mul_node.inputs[0])
                )

            # Look for Add -> Conv pattern
            if node.op == "Add":
                add_node = node
                if len(add_node.outputs[0].outputs) > 1:
                    continue

                # Check for Conv node
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

                # Get the values of the Add and Mul nodes
                constant = self.get_constant_value(add_node, constant_map)
                if constant is None:
                    continue

                add_value, _ = constant

                # Get the weights and bias of the Conv node
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

                # Remove Add nodes
                nodes_to_remove.append(add_node)

                connections_to_fix.append(
                    (add_node.outputs[0], add_node.inputs[0])
                )

        self.graph_cleanup([], nodes_to_remove, connections_to_fix)
        self.onnx_model = gs.export_onnx(self.onnx_gs)

        self.optimize_onnx()

    def fuse_comb_add_mul_to_conv(self) -> None:
        """Fuse combinations of Add and Mul nodes preceding a Conv node directly into
        the Conv node itself.

        The fusion patterns considered are:
        1. Add -> Mul -> Conv
        2. Mul -> Add -> Conv
        """

        nodes_to_remove = []
        connections_to_fix = []

        # Extract constant tensors from the graph
        constant_map = self.get_constant_map(self.onnx_gs)

        # Iterate over the nodes in the graph
        for node in self.onnx_gs.nodes:
            # Look for Mul -> Add -> Conv pattern
            if node.op == "Mul":
                mul_node = node

                # Check for Add node
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

                # Get the values of the Add and Mul nodes
                constant = self.get_constant_value(mul_node, constant_map)
                if constant is None:
                    continue
                mul_value, _ = constant

                constant = self.get_constant_value(add_node, constant_map)
                if constant is None:
                    continue
                add_value, _ = constant

                # Get the weights and bias of the Conv node
                conv_weights = conv_node.inputs[1]
                conv_bias = (
                    conv_node.inputs[2] if len(conv_node.inputs) > 2 else None
                )

                # Adjust the Conv weights by the Mul value on a per-channel basis on axis 1
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

                variable = self.get_variable_input(mul_node)
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

            # Look for Add -> Mul -> Conv pattern
            if node.op == "Add":
                add_node = node

                # Check for Mul node
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

                # Get the values of the Add and Mul nodes
                constant = self.get_constant_value(add_node, constant_map)
                if constant is None:
                    continue
                add_value, _ = constant

                constant = self.get_constant_value(mul_node, constant_map)
                if constant is None:
                    continue
                mul_value, _ = constant

                add_value *= mul_value

                # Get the weights and bias of the Conv node
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

                variable = self.get_variable_input(add_node)
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

        self.graph_cleanup([], nodes_to_remove, connections_to_fix)
        self.onnx_model = gs.export_onnx(self.onnx_gs)

        self.optimize_onnx()

    def fuse_split_concat_to_conv(self) -> None:
        """Fuse Split and Concat nodes that come before a Conv node into the Conv node.

        If any intermediate nodes have channel dimensions, the order of the channels is
        reversed.
        """

        nodes_to_remove = []
        connections_to_fix = []

        # Iterate over the nodes in the graph
        for node in self.onnx_gs.nodes:
            if node.op == "Conv":
                break

            # Look for Split -> Concat -> ... -> Conv pattern
            if node.op == "Split":
                split_node = node

                # Check for Add node
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
                    current_node = next(
                        (n for n in current_node.outputs[0].outputs), None
                    )
                    intermediate_nodes.append(current_node)
                    if current_node is None:
                        break

                conv_node = intermediate_nodes[-1]
                if conv_node.op != "Conv":
                    continue

                conv_weights = conv_node.inputs[1]

                if split_node.attrs["axis"] != concat_node.attrs["axis"]:
                    raise ValueError(
                        f"Split and Concat axis mismatch: {split_node.attrs['axis']} != {concat_node.attrs['axis']}"
                    )

                channels_axis = split_node.attrs["axis"]

                for inter_node in intermediate_nodes[:-1]:
                    constant = self.get_constant_value(
                        inter_node, self.get_constant_map(self.onnx_gs)
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
                        raise ValueError(
                            f"Spatial dimensions mismatch between Conv and intermediate node {inter_node.name}: {constant_value.shape[channels_axis]} != {conv_weights.values.shape[1]}"
                        )

                    # Reverse the order of the constant tensor channels
                    inter_node.inputs[constant_idx].values = np.flip(
                        constant_value, axis=channels_axis
                    )

                # Reverse the order of the channels of the conv node at the axis
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

        self.graph_cleanup([], nodes_to_remove, connections_to_fix)
        self.onnx_model = gs.export_onnx(self.onnx_gs)

        self.optimize_onnx()

    def modify_onnx(self) -> None:
        """Modify the ONNX model by applying a series of optimizations.

        @param passes: List of optimization passes to apply to the ONNX model
        @type passes: Optional[List[str]]
        """

        logger.debug("Substituting Div -> Mul nodes...")
        self.substitute_node_by_type(source_node="Div", target_node="Mul")
        logger.debug("Substituting Sub -> Add nodes...")
        self.substitute_node_by_type(source_node="Sub", target_node="Add")
        logger.debug(
            "Fusing Add and Mul nodes to BatchNormalization nodes and then into Conv nodes..."
        )
        self.fuse_add_mul_to_bn()
        logger.debug("Fusing Add and Mul nodes to Conv nodes...")
        self.fuse_comb_add_mul_to_conv()
        self.fuse_single_add_mul_to_conv()
        logger.debug("Fusing Split and Concat nodes to Conv nodes...")
        self.fuse_split_concat_to_conv()

        self.export_onnx()

    def compare_outputs(self, half: bool = False) -> bool:
        """Compare the outputs of two ONNX models.

        @param half: Flag to use half precision for the input tensors
        @type half: bool
        """

        import onnxruntime as ort

        ort_session_1 = ort.InferenceSession(self.model_path.as_posix())
        ort_session_2 = ort.InferenceSession(self.output_path.as_posix())

        inputs = dict()
        for input in ort_session_1.get_inputs():
            inputs[input.name] = (
                np.random.rand(*input.shape).astype(np.float32)
                if not half
                else np.random.rand(*input.shape).astype(np.float16)
            )

        # Run inference on the first model
        outputs_1 = ort_session_1.run(None, inputs)

        # Run inference on the second model
        outputs_2 = ort_session_2.run(None, inputs)

        equal_outputs = True
        for out1, out2 in zip(outputs_1, outputs_2):
            equal_outputs = equal_outputs and np.allclose(
                out1, out2, rtol=5e-3, atol=5e-3
            )

        return equal_outputs
