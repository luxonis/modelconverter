"""Tests for the RVC4 exporter."""

from pathlib import Path

import onnx
import pytest

from modelconverter.platforms.rvc4.exporter import RVC4Exporter
from modelconverter.utils import PreprocessingEmbeddingError
from modelconverter.utils.config import Config
from modelconverter.utils.types import InputFileType
from tests.helpers.onnx_factory import single_io_onnx


def _make_exporter(
    work_dir: Path,
    mode: str,
    *,
    use_per_row_quantization: bool = False,
) -> RVC4Exporter:
    model = single_io_onnx(work_dir / f"{mode.lower()}.onnx").resolve()

    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": [1, 3, 64, 64],
            "rvc4.quantization_mode": mode,
            "rvc4.use_per_row_quantization": use_per_row_quantization,
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


def test_non_onnx_preprocessing_request_fails(tmp_path: Path):
    onnx_model = single_io_onnx(tmp_path / "model.onnx").resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(onnx_model),
            "shape": [1, 3, 64, 64],
            "encoding": "NONE",
            "mean_values": [1, 2, 3],
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

    with pytest.raises(PreprocessingEmbeddingError, match=r"only embed.*ONNX"):
        RVC4Exporter(stage, output_dir)


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
