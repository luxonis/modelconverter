import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import onnx
import pytest
from onnx import checker, helper
from onnx.onnx_pb import TensorProto

from modelconverter.utils.config import Config
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
    },
    "rvc3": {
        "mo_args": [],
        "compile_tool_args": [],
        "pot_target_device": PotDevice.VPU,
        "disable_calibration": False,
    },
    "rvc4": {
        "snpe_onnx_to_dlc_args": [],
        "snpe_dlc_quant_args": [],
        "snpe_dlc_graph_prepare_args": [],
        "keep_raw_images": False,
        "htp_socs": ["sm8550"],
        "disable_calibration": False,
    },
    "hailo": {
        "optimization_level": 2,
        "compression_level": 2,
        "batch_size": 8,
        "early_stop": False,
        "alls": [],
        "disable_calibration": False,
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
    },
    {
        "name": "output1",
        "data_type": DataType.FLOAT32,
        "shape": [1, 5, 5, 5],
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
    graph = helper.make_graph(
        [], "DummyModel", [input0, input1], [output0, output1]
    )

    model = helper.make_model(graph, producer_name="DummyModelProducer")
    checker.check_model(model)
    onnx.save(model, str(DATA_DIR / "dummy_model.onnx"))
    yield
    shutil.rmtree(DATA_DIR)


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
    Config.clear_instance()
    overrides = {opts[i]: opts[i + 1] for i in range(0, len(opts), 2)}
    config = Config.get_config(path, overrides).model_dump()
    if not multistage:
        name = expected["input_model"].stem
        expected = {
            "name": name,
            "stages": {name: expected},
            "output_dir_name": None,
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
                    "scale_values": [255, 255, 255],
                    "mean_values": [120, 0, 0],
                    "reverse_input_channels": False,
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
                    "data_type": DataType.FLOAT32,
                    "reverse_input_channels": True,
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
                },
            ],
            **DEFAULT_GENERAL_CONFIG,
            "rvc2": {**DEFAULT_TARGET_CONFIGS["rvc2"]},
            "rvc3": {
                "mo_args": ["--compress_to_fp16"],
                "compile_tool_args": [],
                "pot_target_device": PotDevice.VPU,
                "disable_calibration": False,
            },
            "rvc4": {**DEFAULT_TARGET_CONFIGS["rvc4"]},
            "hailo": {
                "disable_calibration": False,
                "optimization_level": 3,
                "compression_level": 3,
                "batch_size": 4,
                "early_stop": False,
                "alls": [],
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
                    "scale_values": [255, 255, 255],
                    "mean_values": [123.675, 116.28, 103.53],
                    "reverse_input_channels": True,
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
                    "scale_values": [255, 255, 255],
                    "mean_values": [123.675, 116.28, 103.53],
                    "reverse_input_channels": True,
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
                    "scale_values": [1.0, 2.0, 3.0],
                    "mean_values": [4.0, 5.0, 6.0],
                    "reverse_input_channels": False,
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
                    "scale_values": [255.0, 255.0, 255.0],
                    "mean_values": [123.675, 116.28, 103.53],
                    "reverse_input_channels": True,
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
                    "scale_values": [1.0, 2.0, 3.0],
                    "mean_values": [4.0, 5.0, 6.0],
                    "reverse_input_channels": False,
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
        Config.clear_instance()
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
        ("reverse_input_channels", "5"),
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
        Config.clear_instance()
        Config.get_config(
            None,
            {
                key: value,
                "input_model": str(DATA_DIR / "dummy_model.onnx"),
            },
        )


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
                    "scale_values": None,
                    "mean_values": None,
                    "reverse_input_channels": True,
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "calibration": DEFAULT_CALIBRATION_CONFIG,
                    "encoding": DEFAULT_ENCODINGS,
                },
                {
                    "name": "input1",
                    "shape": [1, 3, 128, 128],
                    "scale_values": None,
                    "mean_values": None,
                    "reverse_input_channels": True,
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
                    "scale_values": None,
                    "mean_values": None,
                    "reverse_input_channels": True,
                    "data_type": DataType.FLOAT32,
                    "frozen_value": None,
                    "encoding": DEFAULT_ENCODINGS,
                    "calibration": DEFAULT_CALIBRATION_CONFIG,
                },
                {
                    "name": "input1",
                    "shape": [1, 3, 128, 128],
                    "scale_values": None,
                    "mean_values": None,
                    "reverse_input_channels": True,
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
    Config.clear_instance()
    default = Config.get_config(
        DEFAULT_CONFIG_FILE,
        {
            "stages.stage_name.input_model": str(
                DATA_DIR / "dummy_model.onnx"
            ),
            "stages.stage_name.calibration.path": str(CALIBRATION_DATA_DIR_1),
        },
    ).model_dump()
    Config.clear_instance()
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
