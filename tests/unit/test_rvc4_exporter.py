"""Tests for the RVC4 exporter."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto
from PIL import Image

from modelconverter.cli.utils import extract_preprocessing
from modelconverter.platforms.rvc4.exporter import RVC4Exporter
from modelconverter.utils import (
    ModelconverterException,
    PreprocessingEmbeddingError,
)
from modelconverter.utils.config import (
    Config,
    Encodings,
    ImageCalibrationConfig,
    QuantizationOverrides,
    RVC4Config,
)
from modelconverter.utils.preprocessing import reorder_layout
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


def _calibration_exporter(
    tmp_path: Path,
    *,
    quant_args: list[str] | None = None,
    externalize: bool = True,
    disable_calibration: bool = False,
) -> RVC4Exporter:
    model = single_io_onnx(
        tmp_path / "externalized.onnx",
        shape=[1, 3, 1, 1],
        output_shape=[1, 3, 1, 1],
    ).resolve()
    calibration_dir = tmp_path / "externalized-calibration"
    calibration_dir.mkdir(exist_ok=True)
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
                "disable_calibration": disable_calibration,
            },
        },
    )
    if externalize:
        extract_preprocessing(config)
    output_dir = tmp_path / (
        "externalized-output" if externalize else "embedded-output"
    )
    output_dir.mkdir()
    return RVC4Exporter(next(iter(config.stages.values())), output_dir)


def test_externalized_calibration_image_is_written_in_model_domain(
    tmp_path: Path,
):
    exporter = _calibration_exporter(tmp_path)

    input_list = exporter._prepare_calibration_data()

    entry = input_list.read_text().strip()
    raw_path = Path(entry.split(":=", 1)[1])
    actual = np.fromfile(raw_path, dtype=np.float32).reshape(1, 1, 3)
    np.testing.assert_array_equal(
        actual, np.array([[[45.0, 45.0, 45.0]]], dtype=np.float32)
    )


def test_externalized_calibration_matches_embedded_model_output(
    tmp_path: Path,
):
    embedded = _calibration_exporter(tmp_path, externalize=False)
    externalized = _calibration_exporter(tmp_path)
    image_path = tmp_path / "externalized-calibration/pixel.png"

    embedded_input = embedded.inputs["input0"]
    embedded_calibration = embedded_input.calibration
    assert isinstance(embedded_calibration, ImageCalibrationConfig)
    runtime_array, runtime_layout = embedded._read_calibration_file(
        embedded_input, embedded_calibration, image_path
    )
    assert runtime_layout is not None
    runtime_array = reorder_layout(runtime_array, runtime_layout, "NCHW")

    externalized_input = externalized.inputs["input0"]
    externalized_calibration = externalized_input.calibration
    assert isinstance(externalized_calibration, ImageCalibrationConfig)
    model_array, model_layout = externalized._read_calibration_file(
        externalized_input, externalized_calibration, image_path
    )
    assert model_layout is not None
    model_array = reorder_layout(model_array, model_layout, "NCHW")

    def infer(model_path: Path, array: np.ndarray) -> np.ndarray:
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        return np.asarray(
            session.run(None, {session.get_inputs()[0].name: array})[0]
        )

    embedded_output = infer(embedded._input_model, runtime_array)
    externalized_output = infer(externalized._input_model, model_array)

    np.testing.assert_allclose(embedded_output, externalized_output)
    np.testing.assert_array_equal(
        externalized_output,
        np.array([[[[45.0]], [[45.0]], [[45.0]]]], dtype=np.float32),
    )


def test_user_raw_calibration_is_referenced_without_rewriting(tmp_path: Path):
    exporter = _calibration_exporter(tmp_path)
    calibration_dir = tmp_path / "raw-calibration"
    calibration_dir.mkdir()
    source = np.array([[[[1.0]], [[2.0]], [[3.0]]]], dtype=np.float32)
    raw_path = calibration_dir / "model-domain.raw"
    source.tofile(raw_path)
    exporter.inputs["input0"].calibration = ImageCalibrationConfig(
        path=calibration_dir
    )

    input_list = exporter._prepare_calibration_data()

    assert input_list.read_text().strip() == f"input0:={raw_path}"
    np.testing.assert_array_equal(
        np.fromfile(raw_path, dtype=np.float32).reshape(source.shape), source
    )


def test_user_numpy_calibration_is_serialized_without_shape_interpretation(
    tmp_path: Path,
):
    shape = [1, 1, 4, 4]
    model = single_io_onnx(
        tmp_path / "rank-two.onnx", shape=shape, output_shape=shape
    ).resolve()
    calibration_dir = tmp_path / "rank-two-calibration"
    calibration_dir.mkdir()
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.save(calibration_dir / "sample.npy", source)
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": shape,
            "layout": "NCHW",
            "encoding": "NONE",
            "calibration": {"path": str(calibration_dir)},
            "onnx_simplification": False,
            "onnx_optimizations": False,
            "rvc4.quantization_mode": "CUSTOM",
        },
    )
    output_dir = tmp_path / "rank-two-output"
    output_dir.mkdir()
    exporter = RVC4Exporter(next(iter(config.stages.values())), output_dir)

    input_list = exporter._prepare_calibration_data()

    raw_path = Path(input_list.read_text().split(":=", 1)[1].strip())
    np.testing.assert_array_equal(
        np.fromfile(raw_path, dtype=np.float32).reshape(source.shape), source
    )


@pytest.mark.parametrize("joined", [False, True])
def test_externalized_preprocessing_rejects_custom_input_list(
    tmp_path: Path, joined: bool
):
    custom_list = tmp_path / "custom-list.txt"
    custom_list.write_text("input0:=sample.raw\n")
    quant_args = (
        [f"--input_list={custom_list}"]
        if joined
        else ["--input_list", str(custom_list)]
    )
    with pytest.raises(ModelconverterException, match="cannot be used"):
        _calibration_exporter(
            tmp_path,
            quant_args=quant_args,
        )


def test_disabled_calibration_allows_custom_input_list(tmp_path: Path):
    custom_list = tmp_path / "custom-list.txt"
    custom_list.write_text("input0:=sample.raw\n")

    exporter = _calibration_exporter(
        tmp_path,
        quant_args=["--input_list", str(custom_list)],
        disable_calibration=True,
    )

    assert exporter._disable_calibration


def test_identity_externalization_allows_custom_input_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = single_io_onnx(
        tmp_path / "identity.onnx",
        shape=[1, 8],
        output_shape=[1, 8],
    ).resolve()
    calibration_dir = tmp_path / "identity-calibration"
    calibration_dir.mkdir()
    np.save(calibration_dir / "sample.npy", np.zeros((1, 8), np.float32))
    custom_list = tmp_path / "custom-list.txt"
    custom_list.write_text("input0:=sample.raw\n")
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": [1, 8],
            "layout": "NC",
            "encoding": "NONE",
            "calibration": {"path": str(calibration_dir)},
            "onnx_simplification": False,
            "onnx_optimizations": False,
            "rvc4": {
                "quantization_mode": "CUSTOM",
                "snpe_dlc_quant_args": [
                    "--input_list",
                    str(custom_list),
                ],
            },
        },
    )
    extract_preprocessing(config)
    output_dir = tmp_path / "identity-output"
    output_dir.mkdir()
    exporter = RVC4Exporter(next(iter(config.stages.values())), output_dir)
    monkeypatch.setattr(
        exporter, "_subprocess_run", lambda *_args, **_kwargs: None
    )

    result = exporter._calibrate(tmp_path / "model.dlc")

    assert result.name.endswith("-quantized.dlc")


def test_generated_raw_tensor_keeps_configured_layout(tmp_path: Path):
    shape = [1, 2, 3, 4]
    model = single_io_onnx(
        tmp_path / "raw-tensor.onnx",
        shape=shape,
        output_shape=shape,
    ).resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": shape,
            "layout": "NCHW",
            "encoding": "NONE",
            "calibration": {"max_images": 1, "data_type": "float32"},
            "onnx_simplification": False,
            "onnx_optimizations": False,
            "rvc4.quantization_mode": "CUSTOM",
        },
    )
    output_dir = tmp_path / "raw-tensor-output"
    output_dir.mkdir()
    exporter = RVC4Exporter(next(iter(config.stages.values())), output_dir)
    calibration = exporter.inputs["input0"].calibration
    assert isinstance(calibration, ImageCalibrationConfig)
    source = np.load(next(calibration.path.glob("*.npy")))

    input_list = exporter._prepare_calibration_data()

    raw_path = Path(input_list.read_text().split(":=", 1)[1].strip())
    actual = np.fromfile(raw_path, dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(actual, source)


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

    entries = dict(
        token.split(":=", 1) for token in input_list.read_text().split()
    )
    rgb = np.fromfile(entries["rgb_input"], dtype=np.float32)
    bgr = np.fromfile(entries["bgr_input"], dtype=np.float32)
    np.testing.assert_array_equal(rgb, np.array([45, 45, 45], np.float32))
    np.testing.assert_array_equal(bgr, np.array([13, 11, 9], np.float32))
