import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
import pytest
from luxonis_ml.nn_archive.config_building_blocks import PreprocessingBlock
from onnx import checker, helper
from onnx.onnx_pb import TensorProto

from modelconverter.__main__ import extract_preprocessing
from modelconverter.utils.config import Config, EncodingConfig
from modelconverter.utils.nn_archive import (
    modelconverter_config_to_nn,
    process_nn_archive,
)
from modelconverter.utils.onnx_tools import onnx_attach_normalization_to_inputs
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    PotDevice,
    ResizeMethod,
)

DATA_DIR = Path("tests/data/test_utils/test_config")
CALIBRATION_DATA_DIR_1 = DATA_DIR / "calibration_data_1"
CALIBRATION_DATA_DIR_2 = DATA_DIR / "calibration_data_2"
DEFAULT_CONFIG_FILE = "shared_with_container/configs/defaults.yaml"

DEFAULT_ENCODINGS = {
    "from_": Encoding.RGB,
    "to": Encoding.BGR,
}

DEFAULT_TARGET_CONFIGS = {
    "rvc2": {
        "mo_args": [],
        "compile_tool_args": [],
        "superblob": True,
        "number_of_shaves": 8,
        "number_of_cmx_slices": 8,
        "disable_calibration": False,
        "compress_to_fp16": True,
    },
    "rvc3": {
        "mo_args": [],
        "compile_tool_args": [],
        "pot_target_device": PotDevice.VPU,
        "disable_calibration": False,
        "compress_to_fp16": True,
    },
    "rvc4": {
        "snpe_onnx_to_dlc_args": [],
        "snpe_dlc_quant_args": [],
        "snpe_dlc_graph_prepare_args": [],
        "keep_raw_images": False,
        "htp_socs": ["sm8550"],
        "disable_calibration": False,
        "use_per_channel_quantization": True,
        "use_per_row_quantization": False,
    },
    "hailo": {
        "optimization_level": 2,
        "compression_level": 2,
        "batch_size": 8,
        "disable_compilation": False,
        "alls": [],
        "disable_calibration": False,
        "hw_arch": "hailo8",
    },
}

DEFAULT_CALIBRATION_CONFIG = {
    "data_type": DataType.FLOAT32,
    "max_images": 20,
    "max_value": 255,
    "mean": 127.5,
    "min_value": 0,
    "std": 35.0,
}

DEFAULT_GENERAL_CONFIG = {
    "keep_intermediate_outputs": True,
    "disable_onnx_simplification": False,
    "disable_onnx_optimisation": False,
    "output_remote_url": None,
    "put_file_plugin": None,
    "input_bin": None,
    "input_file_type": InputFileType.ONNX,
}

DEFAULT_DUMMY_OUTPUTS = [
    {
        "name": "output0",
        "data_type": DataType.FLOAT32,
        "shape": [1, 10],
        "layout": "NC",
    },
    {
        "name": "output1",
        "data_type": DataType.FLOAT32,
        "shape": [1, 5, 5, 5],
        "layout": "NCDE",
    },
]


@pytest.fixture(scope="module", autouse=True)
def setup():
    CALIBRATION_DATA_DIR_1.mkdir(parents=True, exist_ok=True)
    CALIBRATION_DATA_DIR_2.mkdir(parents=True, exist_ok=True)

    input0 = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    input1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, [1, 3, 128, 128]
    )

    output0 = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 10]
    )
    output1 = helper.make_tensor_value_info(
        "output1", TensorProto.FLOAT, [1, 5, 5, 5]
    )

    shape_tensor = helper.make_tensor(
        name="shape_tensor",
        data_type=TensorProto.INT64,
        dims=[4],
        vals=[1, 5, 5, 5],
    )

    node0 = helper.make_node(
        "Add", inputs=["input0", "input0"], outputs=["intermediate0"]
    )
    node1 = helper.make_node(
        "Add", inputs=["input1", "input1"], outputs=["intermediate1"]
    )
    node2 = helper.make_node(
        "Flatten", inputs=["intermediate0"], outputs=["output0"]
    )
    node3 = helper.make_node(
        "Reshape",
        inputs=["intermediate1", "shape_tensor"],
        outputs=["output1"],
    )

    graph = helper.make_graph(
        [node0, node1, node2, node3],
        "DummyModel",
        [input0, input1],
        [output0, output1],
        initializer=[shape_tensor],
    )

    model = helper.make_model(graph, producer_name="DummyModelProducer")

    checker.check_model(model)
    onnx.save(model, str(DATA_DIR / "dummy_model.onnx"))
    yield
    shutil.rmtree(DATA_DIR)


def set_nested_config_value(
    config: Dict, keys: List[str], values: List[str]
) -> Dict:
    for key, value in zip(keys, values):
        keys = key.split(".")
        current_level = config["model"]

        for k in keys[:-1]:
            if re.match(r"^\d+$", k):
                k = int(k)

            current_level = current_level[k]

        final_key = keys[-1]
        current_level[final_key] = value

    return config


def create_json(
    keys: Optional[List[str]] = None, values: Optional[List[str]] = None
) -> str:
    config = {
        "config_version": "1.0",
        "model": {
            "metadata": {
                "name": "dummy_model",
                "path": "dummy_model.onnx",
                "precision": "float32",
            },
            "inputs": [
                {
                    "name": "input0",
                    "dtype": "float32",
                    "input_type": "image",
                    "shape": [1, 3, 64, 64],
                    "preprocessing": {},
                },
                {
                    "name": "input1",
                    "dtype": "float32",
                    "input_type": "image",
                    "shape": [1, 3, 128, 128],
                    "preprocessing": {},
                },
            ],
            "outputs": [
                {"name": "output0", "dtype": "float32", "shape": [1, 10]},
                {"name": "output1", "dtype": "float32", "shape": [1, 5, 5, 5]},
            ],
            "heads": [],
        },
    }

    if keys and values:
        config = set_nested_config_value(config, keys, values)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(json.dumps(config))
        f.flush()

    return f.name


def create_yaml(append: str = "") -> str:
    config = (
        f"""
input_model: "{DATA_DIR}/dummy_model.onnx"
mean_values: "imagenet"
data_type: float32
shape: [ 1, 3, 256, 256 ]

calibration:
  path: "{CALIBRATION_DATA_DIR_1}"
  max_images: 20

hailo:
  optimization_level: 3
  compression_level: 3
  batch_size: 4

inputs:
  - name: "input0"
    scale_values: [255,255,255]
    shape: [1, 3, 64, 64]
    layout: "NCHW"
    calibration:
      max_images: 100
"""
        + append
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(config)
        f.flush()

    return f.name


def load_and_compare(
    path: Optional[str],
    opts: List[str],
    expected: dict,
    multistage: bool = False,
):
    overrides = {opts[i]: opts[i + 1] for i in range(0, len(opts), 2)}
    config = Config.get_config(path, overrides).model_dump()
    if not multistage:
        name = expected["input_model"].stem
        expected = {
            "name": name,
            "stages": {name: expected},
        }
    assert config == expected


def test_correct():
    path = create_yaml()
    load_and_compare(
        path,
        [
            "mean_values",
            "[120]",
            "inputs.0.name",
            "input0",
            "inputs.1.name",
            "input1",
            "inputs.1.mean_values",
            "[256,256]",
            "inputs.0.mean_values",
            "[120,0,0]",
            "encoding",
            "GRAY",
            "inputs.1.encoding.to",
            "BGR",
            "outputs.0.name",
            "output0",
            "rvc3.mo_args",
            "['--compress_to_fp16']",
        ],
        {
            "input_model": DATA_DIR / "dummy_model.onnx",
            "inputs": [
                {
                    "name": "input0",
                    "shape": [1, 3, 64, 64],
                    "layout": "NCHW",
                    "scale_values": [255, 255, 255],
                    "mean_values": [120, 0, 0],
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "encoding": {
                        "from_": Encoding.GRAY,
                        "to": Encoding.GRAY,
                    },
                    "calibration": {
                        "path": CALIBRATION_DATA_DIR_1,
                        "max_images": 100,
                        "resize_method": ResizeMethod.RESIZE,
                    },
                },
                {
                    "name": "input1",
                    "shape": [1, 3, 256, 256],
                    "layout": "NCHW",
                    "data_type": DataType.FLOAT32,
                    "mean_values": [256, 256],
                    "scale_values": None,
                    "frozen_value": None,
                    "encoding": DEFAULT_ENCODINGS,
                    "calibration": {
                        "path": CALIBRATION_DATA_DIR_1,
                        "max_images": 20,
                        "resize_method": ResizeMethod.RESIZE,
                    },
                },
            ],
            "outputs": [
                {
                    "name": "output0",
                    "data_type": DataType.FLOAT32,
                    "shape": [1, 10],
                    "layout": "NC",
                },
            ],
            **DEFAULT_GENERAL_CONFIG,
            "rvc2": {**DEFAULT_TARGET_CONFIGS["rvc2"]},
            "rvc3": {
                "mo_args": ["--compress_to_fp16"],
                "compile_tool_args": [],
                "pot_target_device": PotDevice.VPU,
                "disable_calibration": False,
                "compress_to_fp16": True,
            },
            "rvc4": {**DEFAULT_TARGET_CONFIGS["rvc4"]},
            "hailo": {
                "disable_calibration": False,
                "optimization_level": 3,
                "compression_level": 3,
                "batch_size": 4,
                "disable_compilation": False,
                "alls": [],
                "hw_arch": "hailo8",
            },
        },
    )


def test_top_level():
    load_and_compare(
        None,
        [
            "input_model",
            str(DATA_DIR / "dummy_model.onnx"),
            "scale_values",
            "[255,255,255]",
            "mean_values",
            "imagenet",
            "calibration.path",
            f"{CALIBRATION_DATA_DIR_1}",
            "calibration.max_images",
            "50",
            "shape",
            "[1,3,64,64]",
            "outputs.0.name",
            "output1",
        ],
        {
            "input_model": DATA_DIR / "dummy_model.onnx",
            "inputs": [
                {
                    "name": "input0",
                    "shape": [1, 3, 64, 64],
                    "layout": "NCHW",
                    "scale_values": [255, 255, 255],
                    "mean_values": [123.675, 116.28, 103.53],
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "encoding": DEFAULT_ENCODINGS,
                    "calibration": {
                        "path": CALIBRATION_DATA_DIR_1,
                        "max_images": 50,
                        "resize_method": ResizeMethod.RESIZE,
                    },
                },
                {
                    "name": "input1",
                    "shape": [1, 3, 64, 64],
                    "layout": "NCHW",
                    "scale_values": [255, 255, 255],
                    "mean_values": [123.675, 116.28, 103.53],
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "encoding": DEFAULT_ENCODINGS,
                    "calibration": {
                        "path": CALIBRATION_DATA_DIR_1,
                        "max_images": 50,
                        "resize_method": ResizeMethod.RESIZE,
                    },
                },
            ],
            "outputs": [
                {
                    "name": "output1",
                    "data_type": DataType.FLOAT32,
                    "shape": [1, 5, 5, 5],
                    "layout": "NCDE",
                },
            ],
            **DEFAULT_GENERAL_CONFIG,
            **DEFAULT_TARGET_CONFIGS,
        },
    )


def test_top_level_override():
    load_and_compare(
        None,
        [
            "input_model",
            str(DATA_DIR / "dummy_model.onnx"),
            "scale_values",
            "[255,255,255]",
            "mean_values",
            "imagenet",
            "calibration.path",
            str(CALIBRATION_DATA_DIR_1),
            "calibration.max_images",
            "50",
            "shape",
            "[1,3,64,64]",
            "outputs.0.name",
            "'883'",
            "outputs.0.shape",
            "[1,10]",
            "outputs.0.data_type",
            "float16",
            "outputs.0.layout",
            "NA",
            "inputs.0.name",
            "input0",
            "inputs.0.shape",
            "[1,3,256,256]",
            "inputs.0.scale_values",
            "[1,2,3]",
            "inputs.0.mean_values",
            "[4,5,6]",
            "inputs.0.data_type",
            "float16",
            "inputs.0.frozen_value",
            "[7,8,9]",
            "inputs.0.calibration.max_images",
            "120",
            "inputs.0.encoding.from",
            "GRAY",
            "inputs.0.calibration.resize_method",
            "CROP",
            "inputs.0.calibration.path",
            str(CALIBRATION_DATA_DIR_2),
            "inputs.1.name",
            "input1",
        ],
        {
            "input_model": DATA_DIR / "dummy_model.onnx",
            "inputs": [
                {
                    "name": "input0",
                    "shape": [1, 3, 256, 256],
                    "layout": "NCHW",
                    "scale_values": [1.0, 2.0, 3.0],
                    "mean_values": [4.0, 5.0, 6.0],
                    "data_type": DataType.FLOAT16,
                    "frozen_value": [7, 8, 9],
                    "encoding": {
                        "from_": Encoding.GRAY,
                        "to": Encoding.GRAY,
                    },
                    "calibration": {
                        "path": CALIBRATION_DATA_DIR_2,
                        "max_images": 120,
                        "resize_method": ResizeMethod.CROP,
                    },
                },
                {
                    "name": "input1",
                    "shape": [1, 3, 64, 64],
                    "layout": "NCHW",
                    "scale_values": [255.0, 255.0, 255.0],
                    "mean_values": [123.675, 116.28, 103.53],
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "encoding": DEFAULT_ENCODINGS,
                    "calibration": {
                        "path": CALIBRATION_DATA_DIR_1,
                        "max_images": 50,
                        "resize_method": ResizeMethod.RESIZE,
                    },
                },
            ],
            "outputs": [
                {
                    "name": "883",
                    "data_type": DataType.FLOAT16,
                    "shape": [1, 10],
                    "layout": "NA",
                },
            ],
            **DEFAULT_GENERAL_CONFIG,
            **DEFAULT_TARGET_CONFIGS,
        },
    )


def test_no_top_level():
    load_and_compare(
        None,
        [
            "input_model",
            str(DATA_DIR / "dummy_model.onnx"),
            "outputs.0.name",
            "output0",
            "inputs.0.name",
            "input0",
            "inputs.0.shape",
            "[1,3,256,256]",
            "inputs.0.scale_values",
            "[1,2,3]",
            "inputs.0.mean_values",
            "[4,5,6]",
            "inputs.0.data_type",
            "float16",
            "inputs.0.frozen_value",
            "[7,8,9]",
            "inputs.0.calibration.max_images",
            "120",
            "inputs.0.encoding.from",
            "GRAY",
            "inputs.0.calibration.resize_method",
            "CROP",
            "inputs.0.calibration.path",
            str(CALIBRATION_DATA_DIR_1),
        ],
        {
            "input_model": DATA_DIR / "dummy_model.onnx",
            "inputs": [
                {
                    "name": "input0",
                    "shape": [1, 3, 256, 256],
                    "layout": "NCHW",
                    "scale_values": [1.0, 2.0, 3.0],
                    "mean_values": [4.0, 5.0, 6.0],
                    "data_type": DataType.FLOAT16,
                    "frozen_value": [7, 8, 9],
                    "encoding": {
                        "from_": Encoding.GRAY,
                        "to": Encoding.GRAY,
                    },
                    "calibration": {
                        "path": CALIBRATION_DATA_DIR_1,
                        "max_images": 120,
                        "resize_method": ResizeMethod.CROP,
                    },
                },
            ],
            "outputs": [
                {
                    "name": "output0",
                    "data_type": DataType.FLOAT32,
                    "shape": [1, 10],
                    "layout": "NC",
                },
            ],
            **DEFAULT_GENERAL_CONFIG,
            **DEFAULT_TARGET_CONFIGS,
        },
    )


@pytest.mark.parametrize(
    "key, value",
    [
        ("non_existent", "value"),
        ("inputs.0.non_existent", "value"),
        ("calibration.non_existent", "value"),
        ("inputs.0.calibration.non_existent", "value"),
    ],
)
def test_missing(key: str, value: str):
    with pytest.raises(ValueError):
        Config.get_config(
            None,
            {
                key: value,
                "input_model": str(DATA_DIR / "dummy_model.onnx"),
            },
        )


@pytest.mark.parametrize(
    "key, value",
    [
        ("inputs.0.encoding.from", "RGBA"),
        ("mean_values", "scale"),
        ("scale_values", "[1,2,dog]"),
        ("rvc2.number_of_shaves", "four"),
        ("hailo.optimization_level", "1000"),
        ("scale_values", "[]"),
    ],
)
def test_incorrect_type(key: str, value: str):
    with pytest.raises(ValueError):
        Config.get_config(
            None,
            {
                key: value,
                "input_model": str(DATA_DIR / "dummy_model.onnx"),
            },
        )


@pytest.mark.parametrize(
    "keys, values",
    [
        (
            [],
            [],
        ),
        (
            ["encoding"],
            ["BGR"],
        ),
        (
            [
                "inputs.0.name",
                "inputs.0.encoding",
                "inputs.1.name",
                "inputs.1.encoding.from",
                "inputs.1.encoding.to",
            ],
            ["input0", "BGR", "input1", "RGB", "BGR"],
        ),
        (
            [
                "inputs.0.name",
                "inputs.0.encoding.from",
                "inputs.1.name",
            ],
            ["input0", "RGB", "input1"],
        ),
        (
            [
                "encoding",
                "mean_values",
                "scale_values",
                "inputs.0.name",
                "inputs.1.name",
                "inputs.1.encoding.from",
                "inputs.1.mean_values",
            ],
            ["BGR", 0, 255, "input0", "input1", "RGB", 127],
        ),
        (
            [
                "inputs.0.name",
                "inputs.0.encoding",
                "inputs.0.mean_values",
                "inputs.0.scale_values",
                "inputs.1.name",
                "inputs.1.encoding.from",
                "inputs.1.encoding.to",
                "inputs.1.mean_values",
                "inputs.1.scale_values",
            ],
            [
                "input0",
                "BGR",
                0,
                1,
                "input1",
                "RGB",
                "BGR",
                [123.675, 116.28, 103.53],
                [58.395, 57.12, 57.375],
            ],
        ),
        (
            [
                "encoding",
                "mean_values",
                "scale_values",
                "inputs.0.name",
                "inputs.0.encoding",
                "inputs.0.mean_values",
                "inputs.0.scale_values",
                "inputs.1.name",
                "inputs.1.encoding.from",
                "inputs.1.encoding.to",
                "inputs.1.mean_values",
                "inputs.1.scale_values",
            ],
            [
                "RGB",
                0,
                255,
                "input0",
                "BGR",
                0,
                1,
                "input1",
                "RGB",
                "BGR",
                [123.675, 116.28, 103.53],
                [58.395, 57.12, 57.375],
            ],
        ),
    ],
)
def test_modified_onnx(keys: List[str], values: List[str]):
    overrides = {keys[i]: values[i] for i in range(len(keys))}
    overrides["input_model"] = str(DATA_DIR / "dummy_model.onnx")
    config = Config.get_config(
        None,
        overrides,
    )
    inputs = next(iter(config.stages.values())).inputs
    input_configs = {inp.name: inp for inp in inputs}

    modified_onnx_path = onnx_attach_normalization_to_inputs(
        DATA_DIR / "dummy_model.onnx",
        DATA_DIR / "dummy_model_modified.onnx",
        input_configs,
        reverse_only=False,
    )
    modified_onnx = onnx.load(modified_onnx_path)

    reverse_inputs = {inp.name: False for inp in inputs}
    normalize_inputs = {inp.name: [False, False] for inp in inputs}
    for node in modified_onnx.graph.node:
        if node.op_type == "Split":
            reverse_inputs[node.input[0]] = True
        elif node.op_type == "Sub":
            inp_name = node.input[1].split("mean_")[1]
            mean_tensor = onnx.numpy_helper.to_array(
                next(
                    t
                    for t in modified_onnx.graph.initializer
                    if t.name == node.input[1]
                )
            )
            assert np.allclose(
                np.squeeze(mean_tensor),
                np.array(input_configs[inp_name].mean_values),
            )
            normalize_inputs[inp_name][0] = True
        elif node.op_type == "Mul":
            inp_name = node.input[1].split("scale_")[1]
            scale_tensor = onnx.numpy_helper.to_array(
                next(
                    t
                    for t in modified_onnx.graph.initializer
                    if t.name == node.input[1]
                )
            )
            assert np.allclose(
                np.squeeze(scale_tensor),
                1 / np.array(input_configs[inp_name].scale_values),
            )
            normalize_inputs[inp_name][1] = True

    for inp, norm in reverse_inputs.items():
        if norm:
            assert input_configs[inp].encoding_mismatch
        else:
            assert not input_configs[inp].encoding_mismatch

    for inp, norm in normalize_inputs.items():
        if norm[0] and norm[1]:
            assert input_configs[inp].mean_values is not None
            assert input_configs[inp].scale_values is not None
        elif norm[0] and not norm[1]:
            assert input_configs[inp].mean_values is not None
            assert input_configs[inp].scale_values is None or [1, 1, 1]
        elif not norm[0] and norm[1]:
            assert input_configs[inp].mean_values is None or [0, 0, 0]
            assert input_configs[inp].scale_values is not None
        else:
            assert input_configs[inp].mean_values is None or [0, 0, 0]
            assert input_configs[inp].scale_values is None or [1, 1, 1]


@pytest.mark.parametrize(
    "keys, values, nn_preprocess, expected",
    [
        (
            [],
            [],
            False,
            [
                {
                    "mean": None,
                    "scale": None,
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
                {
                    "mean": None,
                    "scale": None,
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
            ],
        ),
        (
            [
                "inputs.0.name",
                "inputs.0.encoding",
                "inputs.0.mean_values",
                "inputs.0.scale_values",
                "inputs.1.name",
                "inputs.1.encoding.to",
            ],
            ["input0", "BGR", 127, 255, "input1", "RGB"],
            False,
            [
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
                {
                    "mean": None,
                    "scale": None,
                    "reverse_channels": True,
                    "interleaved_to_planar": False,
                    "dai_type": "RGB888p",
                },
            ],
        ),
        (
            [],
            [],
            True,
            [
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
            ],
        ),
        (
            [
                "inputs.0.name",
                "inputs.0.encoding",
                "inputs.0.mean_values",
                "inputs.0.scale_values",
                "inputs.1.name",
                "inputs.1.encoding.to",
            ],
            ["input0", "BGR", 127, 255, "input1", "RGB"],
            True,
            [
                {
                    "mean": [127, 127, 127],
                    "scale": [255, 255, 255],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": True,
                    "interleaved_to_planar": False,
                    "dai_type": "RGB888p",
                },
            ],
        ),
    ],
)
def test_output_nn_config_from_yaml(
    keys: List[str],
    values: List[str],
    nn_preprocess: bool,
    expected: List[Dict],
):
    overrides = {keys[i]: values[i] for i in range(len(keys))}
    overrides["input_model"] = str(DATA_DIR / "dummy_model.onnx")
    config = Config.get_config(
        None,
        overrides,
    )
    preprocessing = {}
    if nn_preprocess:
        config, preprocessing = extract_preprocessing(config)
    nn_config = modelconverter_config_to_nn(
        config,
        DATA_DIR / "dummy_model.onnx",
        None,
        preprocessing,
        "dummy_model",
        DATA_DIR / "dummy_model.onnx",
    )

    input_0_preprocessing = nn_config.model.inputs[0].preprocessing
    input_1_preprocessing = nn_config.model.inputs[1].preprocessing

    expected_0_preprocessing = PreprocessingBlock(
        mean=expected[0]["mean"],
        scale=expected[0]["scale"],
        reverse_channels=expected[0]["reverse_channels"],
        interleaved_to_planar=expected[0]["interleaved_to_planar"],
        dai_type=expected[0]["dai_type"],
    )
    expected_1_preprocessing = PreprocessingBlock(
        mean=expected[1]["mean"],
        scale=expected[1]["scale"],
        reverse_channels=expected[1]["reverse_channels"],
        interleaved_to_planar=expected[1]["interleaved_to_planar"],
        dai_type=expected[1]["dai_type"],
    )

    assert input_0_preprocessing == expected_0_preprocessing
    assert input_1_preprocessing == expected_1_preprocessing


@pytest.mark.parametrize(
    "keys, values, nn_preprocess, expected",
    [
        (
            [],
            [],
            False,
            [
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
            ],
        ),
        (
            [
                "inputs.0.preprocessing.reverse_channels",
                "inputs.1.preprocessing.reverse_channels",
            ],
            ["False", "True"],
            False,
            [
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
            ],
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
                "inputs.1.preprocessing.dai_type",
            ],
            ["BGR888p", "RGB888p"],
            False,
            [
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
            ],
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
                "inputs.0.preprocessing.reverse_channels",
                "inputs.0.preprocessing.mean",
                "inputs.0.preprocessing.scale",
                "inputs.1.preprocessing.dai_type",
                "inputs.1.preprocessing.reverse_channels",
                "inputs.1.preprocessing.mean",
                "inputs.1.preprocessing.scale",
            ],
            [
                "BGR888p",
                "True",
                [0, 0, 0],
                [255, 255, 255],
                "RGB888p",
                "False",
                [127, 127, 127],
                [255, 255, 255],
            ],
            False,
            [
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
                {
                    "mean": [0, 0, 0],
                    "scale": [1, 1, 1],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
            ],
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
                "inputs.0.preprocessing.reverse_channels",
                "inputs.0.preprocessing.mean",
                "inputs.0.preprocessing.scale",
                "inputs.1.preprocessing.dai_type",
                "inputs.1.preprocessing.reverse_channels",
                "inputs.1.preprocessing.mean",
                "inputs.1.preprocessing.scale",
            ],
            [
                "BGR888p",
                "True",
                [0, 0, 0],
                [255, 255, 255],
                "RGB888p",
                "False",
                [127, 127, 127],
                [255, 255, 255],
            ],
            True,
            [
                {
                    "mean": [0, 0, 0],
                    "scale": [255, 255, 255],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
                {
                    "mean": [127, 127, 127],
                    "scale": [255, 255, 255],
                    "reverse_channels": False,
                    "interleaved_to_planar": False,
                    "dai_type": "BGR888p",
                },
            ],
        ),
    ],
)
def test_output_nn_config_from_nn_archive(
    keys: List[str],
    values: List[str],
    nn_preprocess: bool,
    expected: List[Dict],
):
    nn_archive_path = DATA_DIR / "dummy_model.tar.xz"
    with tempfile.TemporaryDirectory() as tmpdirname:
        tar_path = Path(tmpdirname) / "dummy_model.tar.xz"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(
                str(DATA_DIR / "dummy_model.onnx"), arcname="dummy_model.onnx"
            )
            tar.add(
                create_json(keys=keys, values=values), arcname="config.json"
            )
        shutil.copy(tar_path, nn_archive_path)
    config, archive_cfg, main_stage = process_nn_archive(
        nn_archive_path, overrides=None
    )
    preprocessing = {}
    if nn_preprocess:
        config, preprocessing = extract_preprocessing(config)
    nn_config = modelconverter_config_to_nn(
        config,
        DATA_DIR / "dummy_model.onnx",
        archive_cfg,
        preprocessing,
        main_stage,
        DATA_DIR / "dummy_model.onnx",
    )

    input_0_preprocessing = nn_config.model.inputs[0].preprocessing
    input_1_preprocessing = nn_config.model.inputs[1].preprocessing

    expected_0_preprocessing = PreprocessingBlock(
        mean=expected[0]["mean"],
        scale=expected[0]["scale"],
        reverse_channels=expected[0]["reverse_channels"],
        interleaved_to_planar=expected[0]["interleaved_to_planar"],
        dai_type=expected[0]["dai_type"],
    )
    expected_1_preprocessing = PreprocessingBlock(
        mean=expected[1]["mean"],
        scale=expected[1]["scale"],
        reverse_channels=expected[1]["reverse_channels"],
        interleaved_to_planar=expected[1]["interleaved_to_planar"],
        dai_type=expected[1]["dai_type"],
    )

    assert input_0_preprocessing == expected_0_preprocessing
    assert input_1_preprocessing == expected_1_preprocessing


@pytest.mark.parametrize(
    "key, value, expected",
    [
        (
            "",
            "",
            EncodingConfig(**{"from": Encoding.RGB, "to": Encoding.BGR}),
        ),
        (
            "encoding",
            "NONE",
            EncodingConfig(**{"from": Encoding.NONE, "to": Encoding.NONE}),
        ),
        (
            "encoding",
            "RGB",
            EncodingConfig(**{"from": Encoding.RGB, "to": Encoding.RGB}),
        ),
        (
            "encoding",
            "BGR",
            EncodingConfig(**{"from": Encoding.BGR, "to": Encoding.BGR}),
        ),
        (
            "encoding",
            "GRAY",
            EncodingConfig(**{"from": Encoding.GRAY, "to": Encoding.GRAY}),
        ),
        (
            "encoding.from",
            "BGR",
            EncodingConfig(**{"from": Encoding.BGR, "to": Encoding.BGR}),
        ),
        (
            "encoding.to",
            "RGB",
            EncodingConfig(**{"from": Encoding.RGB, "to": Encoding.RGB}),
        ),
    ],
)
def test_encoding(key: str, value: str, expected: EncodingConfig):
    if key and value:
        config = Config.get_config(
            None,
            {
                key: value,
                "input_model": str(DATA_DIR / "dummy_model.onnx"),
            },
        )
    else:
        config = Config.get_config(
            None,
            {
                "input_model": str(DATA_DIR / "dummy_model.onnx"),
            },
        )
    assert config.get("stages.dummy_model.inputs.0.encoding") == expected
    assert config.get("stages.dummy_model.inputs.1.encoding") == expected


@pytest.mark.parametrize(
    "keys, values, expected",
    [
        (
            [],
            [],
            EncodingConfig(**{"from": Encoding.RGB, "to": Encoding.BGR}),
        ),
        (
            [
                "inputs.0.preprocessing.reverse_channels",
            ],
            ["False"],
            EncodingConfig(**{"from": Encoding.BGR, "to": Encoding.BGR}),
        ),
        (
            [
                "inputs.0.preprocessing.reverse_channels",
            ],
            ["True"],
            EncodingConfig(**{"from": Encoding.RGB, "to": Encoding.BGR}),
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
            ],
            ["BGR888p"],
            EncodingConfig(**{"from": Encoding.BGR, "to": Encoding.BGR}),
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
            ],
            ["RGB888p"],
            EncodingConfig(**{"from": Encoding.RGB, "to": Encoding.BGR}),
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
                "inputs.0.preprocessing.reverse_channels",
            ],
            ["BGR888p", "False"],
            EncodingConfig(**{"from": Encoding.BGR, "to": Encoding.BGR}),
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
                "inputs.0.preprocessing.reverse_channels",
            ],
            ["BGR888p", "True"],
            EncodingConfig(**{"from": Encoding.BGR, "to": Encoding.BGR}),
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
                "inputs.0.preprocessing.reverse_channels",
            ],
            ["RGB888p", "False"],
            EncodingConfig(**{"from": Encoding.RGB, "to": Encoding.BGR}),
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
                "inputs.0.preprocessing.reverse_channels",
            ],
            ["RGB888p", "True"],
            EncodingConfig(**{"from": Encoding.RGB, "to": Encoding.BGR}),
        ),
        (
            [
                "inputs.0.preprocessing.dai_type",
            ],
            ["GRAY8"],
            EncodingConfig(**{"from": Encoding.GRAY, "to": Encoding.GRAY}),
        ),
    ],
)
def test_encoding_nn_archive(
    keys: List[str], values: List[str], expected: EncodingConfig
):
    nn_archive_path = DATA_DIR / "dummy_model.tar.xz"
    with tempfile.TemporaryDirectory() as tmpdirname:
        tar_path = Path(tmpdirname) / "dummy_model.tar.xz"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(
                str(DATA_DIR / "dummy_model.onnx"), arcname="dummy_model.onnx"
            )
            tar.add(
                create_json(keys=keys, values=values), arcname="config.json"
            )
        shutil.copy(tar_path, nn_archive_path)
    config, _, _ = process_nn_archive(nn_archive_path, overrides=None)
    assert config.get("stages.dummy_model.inputs.0.encoding") == expected


def test_onnx_load():
    load_and_compare(
        None,
        [
            "input_model",
            str(DATA_DIR / "dummy_model.onnx"),
            "shape",
            "None",
        ],
        {
            "input_model": DATA_DIR / "dummy_model.onnx",
            "inputs": [
                {
                    "name": "input0",
                    "shape": [1, 3, 64, 64],
                    "layout": "NCHW",
                    "scale_values": None,
                    "mean_values": None,
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "calibration": DEFAULT_CALIBRATION_CONFIG,
                    "encoding": DEFAULT_ENCODINGS,
                },
                {
                    "name": "input1",
                    "shape": [1, 3, 128, 128],
                    "layout": "NCHW",
                    "scale_values": None,
                    "mean_values": None,
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "calibration": DEFAULT_CALIBRATION_CONFIG,
                    "encoding": DEFAULT_ENCODINGS,
                },
            ],
            "outputs": DEFAULT_DUMMY_OUTPUTS,
            **DEFAULT_GENERAL_CONFIG,
            **DEFAULT_TARGET_CONFIGS,
        },
    )


def test_explicit_nones():
    load_and_compare(
        None,
        [
            "input_model",
            str(DATA_DIR / "dummy_model.onnx"),
            "shape",
            "None",
            "output_remote_url",
            "None",
            "mean_values",
            "None",
            "inputs",
            "[]",
            "outputs",
            "[]",
        ],
        {
            "input_model": DATA_DIR / "dummy_model.onnx",
            "input_file_type": InputFileType.ONNX,
            "input_bin": None,
            "inputs": [
                {
                    "name": "input0",
                    "shape": [1, 3, 64, 64],
                    "layout": "NCHW",
                    "scale_values": None,
                    "mean_values": None,
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "encoding": DEFAULT_ENCODINGS,
                    "calibration": DEFAULT_CALIBRATION_CONFIG,
                },
                {
                    "name": "input1",
                    "shape": [1, 3, 128, 128],
                    "layout": "NCHW",
                    "scale_values": None,
                    "mean_values": None,
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "encoding": DEFAULT_ENCODINGS,
                    "calibration": DEFAULT_CALIBRATION_CONFIG,
                },
            ],
            "outputs": DEFAULT_DUMMY_OUTPUTS,
            **DEFAULT_GENERAL_CONFIG,
            **DEFAULT_TARGET_CONFIGS,
        },
    )


def test_defaults():
    default = Config.get_config(
        DEFAULT_CONFIG_FILE,
        {
            "stages.stage_name.input_model": str(
                DATA_DIR / "dummy_model.onnx"
            ),
            "stages.stage_name.calibration.path": str(CALIBRATION_DATA_DIR_1),
        },
    ).model_dump()
    empty = Config.get_config(
        None,
        {
            "stages.stage_name.input_model": str(
                DATA_DIR / "dummy_model.onnx"
            ),
            "stages.stage_name.calibration.path": str(CALIBRATION_DATA_DIR_1),
        },
    ).model_dump()
    assert default == empty
