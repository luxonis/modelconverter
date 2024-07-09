from pathlib import Path
from typing import Dict

import onnx
from onnx import checker, helper
from onnx.onnx_pb import TensorProto

from modelconverter.utils.config import InputConfig

from .exceptions import ONNXException


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

    for input_tensor in graph.input:
        input_name = input_tensor.name
        if input_name not in input_configs:
            raise ONNXException(
                "You either used an invalid input name, or you're attemtpting "
                "to use a hidden network node as an input. This is not supported "
                "in combination with input modifications (mean, scale, etc.). "
                "Either use an actual input name, or modify your network."
            )
        cfg = input_configs[input_name]

        last_output = input_name

        # 1. Reverse channels if needed
        if cfg.reverse_input_channels:
            split_names = [f"split_{i}_{input_name}" for i in range(3)]
            split_node = helper.make_node(
                "Split", inputs=[last_output], outputs=split_names, axis=1
            )
            new_nodes.append(split_node)

            concat_node = helper.make_node(
                "Concat",
                inputs=split_names[::-1],
                outputs=[f"normalized_{input_name}"],
                axis=1,
            )
            new_nodes.append(concat_node)
            last_output = f"normalized_{input_name}"

        # 2. Subtract (mean) if mean_values is not None
        if not reverse_only and cfg.mean_values is not None:
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
                [1, len(cfg.mean_values), 1, 1],
                cfg.mean_values,
            )
            new_initializers.append(mean_tensor)

        # 3. Divide (scale) if scale_values is not None
        if not reverse_only and cfg.scale_values is not None:
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
                [1, len(cfg.scale_values), 1, 1],
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
