"""Host-side preprocessing tests for the OpenVINO exporters."""

from pathlib import Path

import pytest

from modelconverter.platforms.rvc2.exporter import (
    RVC2Exporter,
    _broadcast_preprocessing_values,
)
from modelconverter.platforms.rvc3.exporter import RVC3Exporter
from modelconverter.utils import PreprocessingEmbeddingError
from modelconverter.utils.config import Config
from modelconverter.utils.types import InputFileType
from tests.helpers.onnx_factory import single_io_onnx


def test_scalar_preprocessing_is_expanded_to_resolved_channels():
    assert _broadcast_preprocessing_values([127.0], 2) == [127.0, 127.0]


@pytest.mark.parametrize(
    ("exporter_type", "platform_key"),
    [(RVC2Exporter, "rvc2"), (RVC3Exporter, "rvc3")],
)
def test_raw_two_channel_normalization_is_forwarded_to_model_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exporter_type: type[RVC2Exporter],
    platform_key: str,
):
    shape = [1, 2, 8, 8]
    model = single_io_onnx(
        tmp_path / f"{platform_key}.onnx",
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
            f"{platform_key}.disable_calibration": True,
        },
    )
    output_dir = tmp_path / f"out-{platform_key}"
    output_dir.mkdir()
    exporter = exporter_type(next(iter(config.stages.values())), output_dir)
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
