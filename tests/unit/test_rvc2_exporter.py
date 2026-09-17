"""Host-side preprocessing tests for the OpenVINO exporters."""

from pathlib import Path

import pytest
from onnx import TensorProto

from modelconverter.platforms.rvc2.exporter import RVC2Exporter
from modelconverter.platforms.rvc3.exporter import RVC3Exporter
from modelconverter.utils import (
    Metadata,
    PreprocessingEmbeddingError,
    get_metadata,
)
from modelconverter.utils.config import (
    Config,
    EncodingConfig,
    InputConfig,
    OutputConfig,
    SingleStageConfig,
    broadcast_preprocessing_values,
)
from modelconverter.utils.types import Encoding, InputFileType
from tests.helpers.onnx_factory import build_onnx, single_io_onnx


def _mock_tflite_metadata(
    monkeypatch: pytest.MonkeyPatch,
    input_shapes: dict[str, list[int]],
) -> None:
    """Provide source metadata for mocked TFLite conversion tests."""

    def metadata_for_path(path: Path) -> Metadata:
        if path.suffix == ".tflite":
            return Metadata(input_shapes, {}, {}, {})
        return get_metadata(path)

    monkeypatch.setattr(
        "modelconverter.platforms.rvc2.exporter.get_metadata",
        metadata_for_path,
    )


def test_scalar_preprocessing_is_expanded_to_resolved_channels():
    assert broadcast_preprocessing_values([127.0], 2) == [127.0, 127.0]


@pytest.mark.parametrize(
    ("converted_shape", "expected_shape", "expected_layout"),
    [
        ([1, 4, 8, 8], [1, 4, 8, 8], "NCHW"),
        ([1, 8, 8, 4], [1, 8, 8, 4], "NHWC"),
    ],
)
def test_tflite_raw_layout_tracks_converted_onnx_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    converted_shape: list[int],
    expected_shape: list[int],
    expected_layout: str,
):
    model = tmp_path / "packed.tflite"
    model.touch()
    config = SingleStageConfig.model_construct(
        input_model=model,
        input_file_type=InputFileType.TFLITE,
        inputs=[
            InputConfig(
                name="input0",
                shape=[1, 8, 8, 4],
                layout="NHWC",
                encoding=EncodingConfig.model_validate(
                    {"from": Encoding.NONE, "to": Encoding.NONE}
                ),
            )
        ],
        outputs=[OutputConfig(name="output0", shape=[1], layout="N")],
    )
    output_dir = tmp_path / "out-tflite"
    output_dir.mkdir()
    exporter = RVC2Exporter(config, output_dir)
    monkeypatch.setattr(
        "modelconverter.platforms.rvc2.exporter.tflite2onnx.convert",
        lambda _source, target: single_io_onnx(
            Path(target), shape=converted_shape
        ),
    )
    _mock_tflite_metadata(monkeypatch, {"input0": [1, 8, 8, 4]})

    exporter._transform_tflite_to_onnx()

    inp = exporter.inputs["input0"]
    assert inp.shape == expected_shape
    assert inp.layout == expected_layout


def test_tflite_layout_argument_contains_every_matching_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model = tmp_path / "multi-input.tflite"
    model.touch()
    raw_encoding = EncodingConfig.model_validate(
        {"from": Encoding.NONE, "to": Encoding.NONE}
    )
    config = SingleStageConfig.model_construct(
        input_model=model,
        input_file_type=InputFileType.TFLITE,
        inputs=[
            InputConfig(
                name="image",
                shape=[1, 8, 8, 3],
                layout="NHWC",
                encoding=raw_encoding.model_copy(deep=True),
            ),
            InputConfig(
                name="features",
                shape=[8, 8, 3],
                layout="HWC",
                encoding=raw_encoding.model_copy(deep=True),
            ),
            InputConfig(
                name="unmatched",
                shape=[1, 6, 6, 3],
                layout="NHWC",
                encoding=raw_encoding.model_copy(deep=True),
            ),
        ],
        outputs=[OutputConfig(name="output0", shape=[1], layout="N")],
    )
    output_dir = tmp_path / "out-multi-input"
    output_dir.mkdir()
    exporter = RVC2Exporter(config, output_dir)

    def convert(_source: str, target: str) -> None:
        build_onnx(
            Path(target),
            [
                ("image", [1, 3, 8, 8], TensorProto.FLOAT),
                ("features", [3, 8, 8], TensorProto.FLOAT),
                ("unmatched", [1, 3, 7, 7], TensorProto.FLOAT),
            ],
            [("output0", [1], TensorProto.FLOAT)],
        )

    monkeypatch.setattr(
        "modelconverter.platforms.rvc2.exporter.tflite2onnx.convert", convert
    )
    monkeypatch.setattr(
        "modelconverter.platforms.rvc2.exporter.OV_2021", False
    )
    _mock_tflite_metadata(
        monkeypatch,
        {
            "image": [1, 8, 8, 3],
            "features": [8, 8, 3],
            "unmatched": [1, 6, 6, 3],
        },
    )

    exporter._transform_tflite_to_onnx()

    assert exporter._mo_args == [
        "--layout",
        "image(nchw->nhwc),features(chw->hwc)",
    ]
    assert exporter.inputs["image"].layout == "NCHW"
    assert exporter.inputs["features"].layout == "CHW"
    assert exporter.inputs["unmatched"].shape == [1, 6, 6, 3]
    assert exporter.inputs["unmatched"].layout == "NHWC"


def test_tflite_shape_override_preserves_layout_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model = tmp_path / "model.tflite"
    model.touch()
    config = SingleStageConfig.model_construct(
        input_model=model,
        input_file_type=InputFileType.TFLITE,
        inputs=[
            InputConfig(
                name="data",
                shape=[1, 6, 6, 3],
                layout="NHWC",
                encoding=EncodingConfig.model_validate(
                    {"from": Encoding.NONE, "to": Encoding.NONE}
                ),
            )
        ],
        outputs=[OutputConfig(name="out", shape=[1], layout="N")],
    )
    output_dir = tmp_path / "out-shape-override"
    output_dir.mkdir()
    exporter = RVC2Exporter(config, output_dir)

    monkeypatch.setattr(
        "modelconverter.platforms.rvc2.exporter.tflite2onnx.convert",
        lambda _source, target: single_io_onnx(
            Path(target),
            name="data",
            shape=[1, 3, 8, 8],
            output_name="out",
        ),
    )
    monkeypatch.setattr(
        "modelconverter.platforms.rvc2.exporter.OV_2021", False
    )
    _mock_tflite_metadata(monkeypatch, {"data": [1, 8, 8, 3]})

    exporter._transform_tflite_to_onnx()

    inp = exporter.inputs["data"]
    assert inp.shape == [1, 3, 6, 6]
    assert inp.layout == "NCHW"
    assert exporter._mo_args == ["--layout", "data(nchw->nhwc)"]


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
        PreprocessingEmbeddingError,
        match="existing OpenVINO IR cannot be modified",
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
