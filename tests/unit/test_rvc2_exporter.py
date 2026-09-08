"""Host-side preprocessing tests for the OpenVINO exporters."""

from pathlib import Path

import pytest
from onnx import TensorProto

from modelconverter.platforms.rvc2.exporter import (
    RVC2Exporter,
    _broadcast_preprocessing_values,
)
from modelconverter.platforms.rvc3.exporter import RVC3Exporter
from modelconverter.utils import PreprocessingEmbeddingError
from modelconverter.utils.config import (
    Config,
    EncodingConfig,
    InputConfig,
    OutputConfig,
    SingleStageConfig,
)
from modelconverter.utils.types import Encoding, InputFileType
from tests.helpers.onnx_factory import build_onnx, single_io_onnx


def test_scalar_preprocessing_is_expanded_to_resolved_channels():
    assert _broadcast_preprocessing_values([127.0], 2) == [127.0, 127.0]


def test_raw_two_channel_normalization_is_forwarded_to_model_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    shape = [1, 2, 8, 8]
    model = single_io_onnx(
        tmp_path / "rvc2.onnx",
        shape=shape,
        output_shape=shape,
    ).resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": shape,
            "encoding": "NONE",
            "mean_values": [10, 20],
            "scale_values": [2, 4],
            "onnx_simplification": False,
            "onnx_optimizations": {
                "fuse_add_mul_to_bn": False,
                "fuse_comb_add_mul_to_conv": False,
                "fuse_single_add_mul_to_conv": False,
                "fuse_split_concat_to_conv": False,
                "substitute_sub_with_add": False,
                "substitute_div_with_mul": False,
            },
            "rvc2.disable_calibration": True,
        },
    )
    output_dir = tmp_path / "out-rvc2"
    output_dir.mkdir()
    exporter = RVC2Exporter(next(iter(config.stages.values())), output_dir)
    commands: list[list[str]] = []

    monkeypatch.setattr(
        exporter,
        "_subprocess_run",
        lambda command, **_kwargs: commands.append(command),
    )

    exporter._export_openvino_ir()

    command = commands[0]
    assert command[0] == "mo"
    assert command[command.index("--mean_values") + 1] == "input0[10.0,20.0]"
    assert command[command.index("--scale_values") + 1] == "input0[2.0,4.0]"


def test_selective_bgr_to_rgb_reversal_preserves_runtime_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    shape = [1, 3, 8, 8]
    model = build_onnx(
        tmp_path / "rvc2-mixed-reversal.onnx",
        [
            ("reversed", shape, TensorProto.FLOAT),
            ("unchanged", shape, TensorProto.FLOAT),
        ],
        [
            ("reversed_out", shape, TensorProto.FLOAT),
            ("unchanged_out", shape, TensorProto.FLOAT),
        ],
    ).resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "inputs": [
                {
                    "name": "reversed",
                    "layout": "NCHW",
                    "encoding": {"from": "BGR", "to": "RGB"},
                },
                {
                    "name": "unchanged",
                    "layout": "NCHW",
                    "encoding": "BGR",
                },
            ],
            "onnx_simplification": False,
            "onnx_optimizations": False,
            "rvc2.disable_calibration": True,
        },
    )
    output_dir = tmp_path / "out-rvc2-mixed-reversal"
    output_dir.mkdir()
    exporter = RVC2Exporter(next(iter(config.stages.values())), output_dir)
    monkeypatch.setattr(
        exporter, "_subprocess_run", lambda *_args, **_kwargs: None
    )

    exporter._export_openvino_ir()

    reversed_input = exporter.inputs["reversed"]
    assert reversed_input.encoding.from_ == Encoding.RGB
    assert reversed_input.encoding.to == Encoding.RGB


@pytest.mark.parametrize(
    ("exporter_type", "platform_key"),
    [(RVC2Exporter, "rvc2"), (RVC3Exporter, "rvc3")],
)
def test_existing_ir_with_requested_preprocessing_fails(
    tmp_path: Path,
    exporter_type: type[RVC2Exporter],
    platform_key: str,
):
    shape = [1, 2, 8, 8]
    model = single_io_onnx(
        tmp_path / f"{platform_key}-ir-source.onnx",
        shape=shape,
        output_shape=shape,
    ).resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": shape,
            "encoding": "NONE",
            "mean_values": [10, 20],
            "onnx_simplification": False,
            f"{platform_key}.disable_calibration": True,
        },
    )
    output_dir = tmp_path / f"out-{platform_key}-ir"
    output_dir.mkdir()
    exporter = exporter_type(next(iter(config.stages.values())), output_dir)
    exporter._input_file_type = InputFileType.IR

    with pytest.raises(
        PreprocessingEmbeddingError, match="existing OpenVINO IR"
    ):
        exporter.export()


def test_ir_preprocessing_retry_preserves_unsanitized_bin_source(
    tmp_path: Path,
):
    xml_path = tmp_path / "model with spaces.xml"
    bin_path = tmp_path / "model with spaces.bin"
    xml_path.write_text("<net/>")
    bin_path.write_bytes(b"weights")
    config = SingleStageConfig.model_construct(
        input_model=xml_path,
        input_bin=bin_path,
        input_file_type=InputFileType.IR,
        inputs=[
            InputConfig(
                name="input0",
                shape=[1, 2, 8, 8],
                layout="NCHW",
                encoding=EncodingConfig.model_validate(
                    {"from": Encoding.NONE, "to": Encoding.NONE}
                ),
                mean_values=[10.0, 20.0],
            )
        ],
        outputs=[OutputConfig(name="output0", shape=[1], layout="N")],
    )
    output_dir = tmp_path / "out-ir-retry"
    output_dir.mkdir()

    first_exporter = RVC2Exporter(config, output_dir)
    with pytest.raises(
        PreprocessingEmbeddingError, match="existing OpenVINO IR"
    ):
        first_exporter.export()

    assert config.input_bin == bin_path
    config.inputs[0].mean_values = None

    RVC2Exporter(config, output_dir)

    assert (
        output_dir / "intermediate_outputs" / "model_with_spaces.bin"
    ).read_bytes() == b"weights"
