"""Tests for the RVC4 exporter."""

import json
from pathlib import Path

import pytest

from modelconverter.platforms.rvc4.exporter import RVC4Exporter
from modelconverter.utils.config import (
    Config,
    Encodings,
    QuantizationOverrides,
    RVC4Config,
)
from tests.helpers.onnx_factory import single_io_onnx


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


def test_path_quantization_overrides_passthrough_without_io_normalization(
    work_dir: Path,
):
    exporter = _make_exporter(
        work_dir,
        "CUSTOM",
        normalize_io_encodings=False,
    )
    encodings_path = work_dir / "encodings.json"
    raw = (
        '{"top_level_vendor_key":true,\n'
        '"activation_encodings":{"input0":[{"bitwidth":16,"vendor_key":1}]},\n'
        '"param_encodings":{"weight":[{"bitwidth":8,"param_vendor_key":2}]}}\n'
    )
    encodings_path.write_text(raw)

    generated = exporter._generate_io_encodings(
        QuantizationOverrides.from_path(encodings_path)
    )

    assert generated == encodings_path
    assert generated.read_text() == raw


def test_inline_quantization_overrides_preserved_without_io_normalization(
    work_dir: Path,
):
    exporter = _make_exporter(
        work_dir,
        "CUSTOM",
        normalize_io_encodings=False,
    )
    payload = {
        "top_level_vendor_key": {"keep": True},
        "activation_encodings": {
            "input0": [{"bitwidth": 16, "activation_vendor_key": [1]}],
        },
        "param_encodings": {
            "weight": [{"bitwidth": 8, "param_vendor_key": {"x": 2}}],
        },
    }

    generated = exporter._generate_io_encodings(
        QuantizationOverrides.from_payload(payload)
    )

    assert json.loads(generated.read_text()) == payload


def test_io_normalization_preserves_unrelated_override_data(work_dir: Path):
    exporter = _make_exporter(work_dir, "CUSTOM")
    payload = {
        "top_level_vendor_key": {"keep": True},
        "activation_encodings": {
            "input0": [
                {
                    "bitwidth": 16,
                    "dtype": "float",
                    "scale": 0.5,
                    "activation_vendor_key": [1],
                }
            ],
            "hidden": [
                {
                    "bitwidth": 16,
                    "dtype": "int",
                    "hidden_vendor_key": "keep",
                }
            ],
            "output0": [{"output_vendor_key": "keep"}],
        },
        "param_encodings": {
            "weight": [{"bitwidth": 8, "param_vendor_key": {"x": 2}}],
        },
    }

    generated = exporter._generate_io_encodings(
        QuantizationOverrides.from_payload(payload)
    )
    generated_payload = json.loads(generated.read_text())

    assert generated_payload["top_level_vendor_key"] == {"keep": True}
    assert generated_payload["activation_encodings"]["input0"] == [
        {"activation_vendor_key": [1], "bitwidth": 8, "dtype": "int"}
    ]
    assert generated_payload["activation_encodings"]["output0"] == [
        {"output_vendor_key": "keep", "bitwidth": 8, "dtype": "int"}
    ]
    assert generated_payload["activation_encodings"]["hidden"] == [
        {
            "bitwidth": 16,
            "dtype": "int",
            "hidden_vendor_key": "keep",
        }
    ]
    assert generated_payload["param_encodings"] == payload["param_encodings"]


def test_io_normalization_preserves_multiple_exposed_entries(
    work_dir: Path,
):
    exporter = _make_exporter(work_dir, "CUSTOM")
    payload = {
        "activation_encodings": {
            "input0": [
                {
                    "bitwidth": 16,
                    "scale": 0.5,
                    "first_vendor_key": "first",
                },
                {
                    "dtype": "float",
                    "offset": 2,
                    "second_vendor_key": "second",
                },
            ],
        },
        "param_encodings": {},
    }

    generated = exporter._generate_io_encodings(
        QuantizationOverrides.from_payload(payload)
    )
    generated_payload = json.loads(generated.read_text())

    assert generated_payload["activation_encodings"]["input0"] == [
        {"first_vendor_key": "first", "bitwidth": 8, "dtype": "int"},
        {"second_vendor_key": "second", "bitwidth": 8, "dtype": "int"},
    ]


def test_io_normalization_populates_empty_exposed_list(
    work_dir: Path,
):
    exporter = _make_exporter(work_dir, "CUSTOM")
    payload = {
        "activation_encodings": {
            "input0": [],
        },
        "param_encodings": {},
    }

    generated = exporter._generate_io_encodings(
        QuantizationOverrides.from_payload(payload)
    )
    generated_payload = json.loads(generated.read_text())

    assert generated_payload["activation_encodings"]["input0"] == [
        {"bitwidth": 8, "dtype": "int"}
    ]


def test_io_normalization_preserves_dict_shaped_exposed_entry(
    work_dir: Path,
):
    exporter = _make_exporter(work_dir, "CUSTOM")
    payload = {
        "activation_encodings": {
            "input0": {
                "name": "input0",
                "bitwidth": 16,
                "scale": 0.5,
                "vendor_key": "keep",
            },
        },
        "param_encodings": {},
    }

    generated = exporter._generate_io_encodings(
        QuantizationOverrides.from_payload(payload)
    )
    generated_payload = json.loads(generated.read_text())

    assert generated_payload["activation_encodings"]["input0"] == {
        "name": "input0",
        "vendor_key": "keep",
        "bitwidth": 8,
        "dtype": "int",
    }


def test_io_normalization_preserves_flat_activation_list_shape(
    work_dir: Path,
):
    exporter = _make_exporter(work_dir, "CUSTOM")
    payload = {
        "activation_encodings": [
            {
                "name": "hidden",
                "bitwidth": 16,
                "hidden_vendor_key": "keep",
            },
            {
                "name": "input0",
                "scale": 0.5,
                "input_vendor_key": "keep",
            },
        ],
        "param_encodings": {},
    }

    generated = exporter._generate_io_encodings(
        QuantizationOverrides.from_payload(payload)
    )
    generated_payload = json.loads(generated.read_text())

    assert generated_payload["activation_encodings"] == [
        {
            "name": "hidden",
            "bitwidth": 16,
            "hidden_vendor_key": "keep",
        },
        {
            "name": "input0",
            "input_vendor_key": "keep",
            "bitwidth": 8,
            "dtype": "int",
        },
        {"name": "output0", "bitwidth": 8, "dtype": "int"},
    ]


@pytest.mark.parametrize(
    "entry",
    [
        {"bitwidth": 16},
        {"name": ""},
        {"name": 123},
    ],
)
def test_io_normalization_rejects_invalid_flat_activation_name(
    work_dir: Path,
    entry: dict[str, int | str],
):
    exporter = _make_exporter(work_dir, "CUSTOM")
    payload = {
        "activation_encodings": [entry],
        "param_encodings": {},
    }

    with pytest.raises(
        TypeError,
        match=r"nonempty string `name`",
    ):
        exporter._generate_io_encodings(
            QuantizationOverrides.from_payload(payload)
        )


def test_io_normalization_rejects_unsupported_activation_group_shape(
    work_dir: Path,
):
    exporter = _make_exporter(work_dir, "CUSTOM")
    payload = {"activation_encodings": "raw-string", "param_encodings": {}}

    with pytest.raises(TypeError, match="normalize_io_encodings=False"):
        exporter._generate_io_encodings(
            QuantizationOverrides.from_payload(payload)
        )


def test_io_normalization_rejects_unsupported_exposed_entry_shape(
    work_dir: Path,
):
    exporter = _make_exporter(work_dir, "CUSTOM")
    payload = {
        "activation_encodings": {"input0": ["not-a-dict"]},
        "param_encodings": {},
    }

    with pytest.raises(TypeError, match=r"activation_encodings\.input0"):
        exporter._generate_io_encodings(
            QuantizationOverrides.from_payload(payload)
        )


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
