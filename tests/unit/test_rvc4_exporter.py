"""Tests for the RVC4 exporter."""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto
from PIL import Image

from modelconverter.cli.utils import extract_preprocessing
from modelconverter.platforms.rvc4.exporter import RVC4Exporter
from modelconverter.utils import (
    ModelconverterException,
    PreprocessingEmbeddingError,
)
from modelconverter.utils.config import Config, Encodings, RVC4Config
from modelconverter.utils.types import InputFileType
from tests.helpers.onnx_factory import build_onnx, single_io_onnx


def _make_exporter(
    work_dir: Path,
    mode: str,
    *,
    use_per_row_quantization: bool = False,
    normalize_io_encodings: bool = True,
) -> RVC4Exporter:
    model = single_io_onnx(work_dir / f"{mode.lower()}.onnx").resolve()

    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": [1, 3, 64, 64],
            "rvc4.quantization_mode": mode,
            "rvc4.use_per_row_quantization": use_per_row_quantization,
            "rvc4.normalize_io_encodings": normalize_io_encodings,
        },
    )
    stage = next(iter(config.stages.values()))

    output_dir = (
        work_dir / f"out-{mode.lower()}-{int(use_per_row_quantization)}"
    )
    output_dir.mkdir()

    exporter = RVC4Exporter(stage, output_dir)

    # This is deliberately true in the source configuration. The INT16
    # contract must suppress the generic default at command construction time,
    # without changing the default for existing RVC4 modes.
    assert exporter._use_per_channel_quantization

    return exporter


def _capture_quant_command(
    exporter: RVC4Exporter,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    input_list = work_dir / "input_list.txt"
    input_list.write_text("dummy.raw\n")

    monkeypatch.setattr(
        exporter,
        "_prepare_calibration_data",
        lambda: input_list,
    )

    calls: list[tuple[list[str], str | None]] = []

    def fake_subprocess_run(
        command: list[str],
        meta_name: str | None = None,
    ) -> None:
        calls.append((command, meta_name))

    monkeypatch.setattr(
        exporter,
        "_subprocess_run",
        fake_subprocess_run,
    )

    input_dlc = work_dir / "input.dlc"
    input_dlc.touch()

    exporter._calibrate(input_dlc)

    quant_calls = [
        command
        for command, meta_name in calls
        if meta_name == "quantization_cmd"
    ]
    assert len(quant_calls) == 1

    command = quant_calls[0]
    assert command[0] == "snpe-dlc-quant"
    return command


def _flag_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_normalize_io_encodings_default_true():
    assert RVC4Config().normalize_io_encodings is True


@pytest.mark.parametrize("normalize_io_encodings", [True, False])
def test_normalize_io_encodings_controls_exposed_tensor_rewrite(
    work_dir: Path,
    normalize_io_encodings: bool,
):
    exporter = _make_exporter(
        work_dir,
        "CUSTOM",
        normalize_io_encodings=normalize_io_encodings,
    )
    assert exporter._normalize_io_encodings is normalize_io_encodings

    custom_encoding = {"bitwidth": 16, "dtype": "int"}
    encodings = Encodings.model_validate(
        {
            "activation_encodings": {
                "input0": [custom_encoding],
                "hidden": [custom_encoding],
                "output0": [custom_encoding],
            },
            "param_encodings": {
                "weight": [custom_encoding],
            },
        }
    )

    encodings_path = exporter._generate_io_encodings(encodings)
    generated = json.loads(encodings_path.read_text())

    expected_io_encoding = (
        [{"bitwidth": 8, "dtype": "int"}]
        if normalize_io_encodings
        else [custom_encoding]
    )

    assert generated["activation_encodings"]["input0"] == expected_io_encoding
    assert generated["activation_encodings"]["output0"] == expected_io_encoding
    assert generated["activation_encodings"]["hidden"] == [custom_encoding]
    assert generated["param_encodings"]["weight"] == [custom_encoding]


def test_int16_standard_native_quantizer_contract(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = _make_exporter(work_dir, "INT16_STANDARD")
    command = _capture_quant_command(exporter, work_dir, monkeypatch)

    assert _flag_value(command, "--weights_bitwidth") == "16"
    assert _flag_value(command, "--act_bitwidth") == "16"

    assert "--use_per_channel_quantization" not in command
    assert "--use_per_row_quantization" not in command

    assert "--bias_bitwidth" not in command
    assert "--override_params" not in command


def test_int8_standard_keeps_per_channel_default(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = _make_exporter(work_dir, "INT8_STANDARD")
    command = _capture_quant_command(exporter, work_dir, monkeypatch)

    assert "--use_per_channel_quantization" in command
    assert "--weights_bitwidth" not in command
    assert "--act_bitwidth" not in command


def test_int16_standard_does_not_change_per_row_behavior(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    exporter = _make_exporter(
        work_dir,
        "INT16_STANDARD",
        use_per_row_quantization=True,
    )
    command = _capture_quant_command(exporter, work_dir, monkeypatch)

    assert "--use_per_channel_quantization" not in command
    assert "--use_per_row_quantization" in command


def test_non_onnx_default_preprocessing_error_names_encoding_remedy(
    tmp_path: Path,
):
    onnx_model = single_io_onnx(tmp_path / "model.onnx").resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(onnx_model),
            "shape": [1, 3, 64, 64],
            "rvc4.disable_calibration": True,
        },
    )
    model = tmp_path / "model.tflite"
    model.write_bytes(b"TFL3")
    stage = next(iter(config.stages.values()))
    stage.input_model = model
    stage.input_file_type = InputFileType.TFLITE
    output_dir = tmp_path / "out-tflite"
    output_dir.mkdir()

    with pytest.raises(
        PreprocessingEmbeddingError,
        match=r"`encoding RGB` or `encoding NONE`",
    ):
        RVC4Exporter(stage, output_dir)


def test_non_onnx_without_preprocessing_is_accepted(tmp_path: Path):
    onnx_model = single_io_onnx(tmp_path / "source.onnx").resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(onnx_model),
            "shape": [1, 3, 64, 64],
            "encoding": "RGB",
            "rvc4.disable_calibration": True,
        },
    )
    model = tmp_path / "model.tflite"
    model.write_bytes(b"TFL3")
    stage = next(iter(config.stages.values()))
    stage.input_model = model
    stage.input_file_type = InputFileType.TFLITE
    output_dir = tmp_path / "out-tflite-no-preprocessing"
    output_dir.mkdir()

    exporter = RVC4Exporter(stage, output_dir)

    assert exporter.inputs["input0"].requires_input_preprocessing() is False


def test_two_channel_normalization_is_embedded(tmp_path: Path):
    shape = [1, 2, 8, 8]
    model = single_io_onnx(
        tmp_path / "two-channel.onnx",
        shape=shape,
        output_shape=shape,
    ).resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": shape,
            "encoding": "NONE",
            "mean_values": [113.55, 113.55],
            "scale_values": [68.646, 68.646],
            "onnx_simplification": False,
            "onnx_optimizations": {
                "fuse_add_mul_to_bn": False,
                "fuse_comb_add_mul_to_conv": False,
                "fuse_single_add_mul_to_conv": False,
                "fuse_split_concat_to_conv": False,
                "substitute_sub_with_add": False,
                "substitute_div_with_mul": False,
            },
            "rvc4.disable_calibration": True,
        },
    )
    output_dir = tmp_path / "out-two-channel"
    output_dir.mkdir()

    exporter = RVC4Exporter(next(iter(config.stages.values())), output_dir)

    operations = [
        node.op_type for node in onnx.load(exporter._input_model).graph.node
    ]
    assert operations[:2] == ["Sub", "Mul"]


def _externalized_calibration_exporter(
    tmp_path: Path,
    *,
    quant_args: list[str] | None = None,
) -> RVC4Exporter:
    model = single_io_onnx(
        tmp_path / "externalized.onnx",
        shape=[1, 3, 1, 1],
        output_shape=[1, 3, 1, 1],
    ).resolve()
    calibration_dir = tmp_path / "externalized-calibration"
    calibration_dir.mkdir()
    Image.fromarray(np.array([[[100, 110, 120]]], dtype=np.uint8)).save(
        calibration_dir / "pixel.png"
    )
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": [1, 3, 1, 1],
            "layout": "NCHW",
            "encoding": {"from": "RGB", "to": "BGR"},
            "mean_values": [10, 20, 30],
            "scale_values": 2,
            "calibration": {"path": str(calibration_dir)},
            "onnx_simplification": False,
            "onnx_optimizations": False,
            "rvc4": {
                "quantization_mode": "CUSTOM",
                "snpe_dlc_quant_args": quant_args or [],
            },
        },
    )
    extract_preprocessing(config)
    output_dir = tmp_path / "externalized-output"
    output_dir.mkdir()
    return RVC4Exporter(next(iter(config.stages.values())), output_dir)


def test_externalized_calibration_image_is_written_in_model_domain(
    tmp_path: Path,
):
    exporter = _externalized_calibration_exporter(tmp_path)

    input_list = exporter._prepare_calibration_data()

    entry = input_list.read_text().strip()
    raw_path = Path(entry.split(":=", 1)[1])
    actual = np.fromfile(raw_path, dtype=np.float32).reshape(1, 1, 3)
    np.testing.assert_array_equal(
        actual, np.array([[[45.0, 45.0, 45.0]]], dtype=np.float32)
    )


def test_externalized_preprocessing_rejects_custom_input_list(tmp_path: Path):
    custom_list = tmp_path / "custom-list.txt"
    custom_list.write_text("input0:=sample.raw\n")
    exporter = _externalized_calibration_exporter(
        tmp_path,
        quant_args=["--input_list", str(custom_list)],
    )

    with pytest.raises(ModelconverterException, match="cannot be used"):
        exporter._calibrate(tmp_path / "model.dlc")


def test_externalized_multi_input_calibration_uses_each_input_contract(
    tmp_path: Path,
) -> None:
    shape = [1, 3, 1, 1]
    model = build_onnx(
        tmp_path / "multi-input.onnx",
        [
            ("rgb_input", shape, TensorProto.FLOAT),
            ("bgr_input", shape, TensorProto.FLOAT),
        ],
        [
            ("rgb_output", shape, TensorProto.FLOAT),
            ("bgr_output", shape, TensorProto.FLOAT),
        ],
    ).resolve()
    rgb_dir = tmp_path / "rgb-calibration"
    bgr_dir = tmp_path / "bgr-calibration"
    rgb_dir.mkdir()
    bgr_dir.mkdir()
    Image.fromarray(np.array([[[100, 110, 120]]], dtype=np.uint8)).save(
        rgb_dir / "pixel.png"
    )
    Image.fromarray(np.array([[[50, 60, 70]]], dtype=np.uint8)).save(
        bgr_dir / "pixel.png"
    )
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "inputs": [
                {
                    "name": "rgb_input",
                    "layout": "NCHW",
                    "encoding": {"from": "RGB", "to": "BGR"},
                    "mean_values": [10, 20, 30],
                    "scale_values": 2,
                    "calibration": {"path": str(rgb_dir)},
                },
                {
                    "name": "bgr_input",
                    "layout": "NCHW",
                    "encoding": "BGR",
                    "mean_values": 5,
                    "scale_values": 5,
                    "calibration": {"path": str(bgr_dir)},
                },
            ],
            "onnx_simplification": False,
            "onnx_optimizations": False,
            "rvc4.quantization_mode": "CUSTOM",
        },
    )
    extract_preprocessing(config)
    output_dir = tmp_path / "multi-input-output"
    output_dir.mkdir()
    exporter = RVC4Exporter(next(iter(config.stages.values())), output_dir)

    input_list = exporter._prepare_calibration_data()

    entries = {
        name: Path(path)
        for token in input_list.read_text().split()
        for name, path in [token.split(":=", 1)]
    }
    rgb = np.fromfile(entries["rgb_input"], dtype=np.float32)
    bgr = np.fromfile(entries["bgr_input"], dtype=np.float32)
    np.testing.assert_array_equal(rgb, np.array([45, 45, 45], np.float32))
    np.testing.assert_array_equal(bgr, np.array([13, 11, 9], np.float32))
