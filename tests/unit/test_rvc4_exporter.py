"""Tests for the RVC4 exporter."""

from pathlib import Path

import pytest

from modelconverter.platforms.rvc4.exporter import RVC4Exporter
from modelconverter.utils.config import Config
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
