"""Host-side unit tests for ``modelconverter.utils.config``.

No network, cloud, Docker or vendor tooling. The sub-config validators are
exercised both directly, by constructing the pydantic models, and through the
full ``Config.get_config`` pipeline.
"""

import json
from pathlib import Path

import onnx
import pytest
from luxonis_ml.typing import ParamValue
from onnx import TensorProto, checker, helper
from pydantic import ValidationError

from modelconverter.utils import config as config_module
from modelconverter.utils.config import (
    Config,
    EncodingConfig,
    Encodings,
    HailoConfig,
    ImageCalibrationConfig,
    InputConfig,
    LinkCalibrationConfig,
    ONNXOptimizationsConfig,
    OutputConfig,
    RandomCalibrationConfig,
    RVC2Config,
    RVC3Config,
    RVC4Config,
    SingleStageConfig,
    _extract_bin_xml_from_ir,
    _get_onnx_inter_info,
    _get_onnx_node_info,
    _get_onnx_tensor_info,
    generate_renamed_onnx,
)
from modelconverter.utils.constants import MISC_DIR, MODELS_DIR
from modelconverter.utils.metadata import Metadata
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    Platform,
    QuantizationMode,
    ResizeMethod,
)
from tests.helpers.onnx_factory import (
    dynamic_batch_onnx,
    intermediate_info_onnx,
    standard_dummy_onnx,
)


def _input_config(**data: ParamValue) -> InputConfig:
    """Validate raw user configuration through Pydantic's public API."""
    return InputConfig.model_validate(data)


def _rvc4_config(**data: ParamValue) -> RVC4Config:
    """Validate raw user configuration through Pydantic's public API."""
    return RVC4Config.model_validate(data)


def _models_dir() -> Path:
    """The models directory of the cache, re-rooted into the temp dir by
    ``_isolate_cwd``."""
    return MODELS_DIR


def _dummy(name: str = "dummy_model.onnx") -> Path:
    """Absolute path to a freshly built two-in/two-out dummy model."""
    return standard_dummy_onnx(_models_dir() / name).resolve()


def _single_stage(config: Config) -> SingleStageConfig:
    return next(iter(config.stages.values()))


def _named_node_onnx(path: Path) -> Path:
    """A model whose node *name* differs from its output *tensor* name.

    ``mynode`` produces tensor ``y`` (described in ``value_info``), so a
    lookup by node name misses the tensor search and falls through to
    the node search -- the branch the standard fixtures cannot reach.
    """
    inp = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [1, 3, 8, 8]
    )
    out = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 3, 8, 8]
    )
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    node0 = helper.make_node("Relu", ["input0"], ["y"], name="mynode")
    node1 = helper.make_node("Identity", ["y"], ["output0"], name="outnode")
    graph = helper.make_graph(
        [node0, node1], "NamedNode", [inp], [out], value_info=[y]
    )
    model = helper.make_model(graph, producer_name="DummyModelProducer")
    checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _pt_model() -> Path:
    pt = (_models_dir() / "yolo.pt").resolve()
    pt.write_bytes(b"not-a-real-checkpoint")
    return pt


def test_flat_config_wrapped_and_renamed():
    """A flat single-stage config is wrapped and the ``default_stage``
    placeholder is renamed to the input-model stem."""
    dummy = _dummy()
    config = Config.get_config(None, {"input_model": str(dummy)})
    assert config.name == dummy.stem
    assert set(config.stages) == {dummy.stem}
    assert "default_stage" not in config.stages


def test_flat_config_with_explicit_name_kept():
    """An explicit name suppresses the ``default_stage`` rename."""
    dummy = _dummy()
    config = Config.get_config(
        None, {"input_model": str(dummy), "name": "custom"}
    )
    assert config.name == "custom"
    assert set(config.stages) == {"custom"}


def test_multistage_extras_fanned_into_each_stage():
    """Top-level extras are distributed to stages missing them, and the
    derived name joins the stage keys."""
    dummy = _dummy()
    config = Config.get_config(
        None,
        {
            "stages.s1.input_model": str(dummy),
            "stages.s2.input_model": str(dummy),
            "shape": "[1, 3, 8, 8]",
            "stages.s1.shape": "[1, 3, 16, 16]",
        },
    )
    assert config.name == "s1-s2"
    # s1 kept its own shape, s2 inherited the top-level extra.
    assert config.stages["s1"].inputs[0].shape == [1, 3, 16, 16]
    assert config.stages["s2"].inputs[0].shape == [1, 3, 8, 8]


def test_multistage_explicit_name_preserved():
    dummy = _dummy()
    config = Config.get_config(
        None,
        {
            "name": "pipeline",
            "stages.s1.input_model": str(dummy),
            "stages.s2.input_model": str(dummy),
        },
    )
    assert config.name == "pipeline"


def test_get_stage_config_none_with_single_stage():
    config = Config.get_config(None, {"input_model": str(_dummy())})
    assert config.get_stage_config(None) is _single_stage(config)


def test_get_stage_config_none_with_multiple_stages_raises():
    dummy = _dummy()
    config = Config.get_config(
        None,
        {
            "stages.s1.input_model": str(dummy),
            "stages.s2.input_model": str(dummy),
        },
    )
    with pytest.raises(ValueError, match="Multiple stages"):
        config.get_stage_config(None)


def test_get_stage_config_explicit_key():
    dummy = _dummy()
    config = Config.get_config(
        None,
        {
            "stages.s1.input_model": str(dummy),
            "stages.s2.input_model": str(dummy),
        },
    )
    assert config.get_stage_config("s2") is config.stages["s2"]


def test_missing_input_model_raises():
    with pytest.raises(ValueError, match="`input_model` must be provided"):
        Config.get_config(None, {})


def test_onnx_path_resolved_absolute():
    dummy = _dummy()
    config = Config.get_config(None, {"input_model": str(dummy)})
    stage = _single_stage(config)
    assert stage.input_model == dummy
    assert stage.input_file_type == InputFileType.ONNX
    assert stage.input_bin is None


def test_ir_path_extracts_bin_and_xml(monkeypatch: pytest.MonkeyPatch):
    """The IR branch splits the ``.xml``/``.bin`` pair; ``get_metadata``
    is stubbed so no OpenVINO runtime is needed."""
    xml = (_models_dir() / "model.xml").resolve()
    bin_ = (_models_dir() / "model.bin").resolve()
    xml.write_text("<net/>")
    bin_.write_bytes(b"\x00")

    def fake_metadata(path: Path) -> Metadata:
        return Metadata(
            input_shapes={"input0": [1, 3, 8, 8]},
            input_dtypes={"input0": DataType.FLOAT32},
            output_shapes={"output0": [1, 8]},
            output_dtypes={"output0": DataType.FLOAT32},
        )

    monkeypatch.setattr(config_module, "get_metadata", fake_metadata)
    config = Config.get_config(None, {"input_model": str(xml)})
    stage = _single_stage(config)
    assert stage.input_model == xml
    assert stage.input_bin == bin_
    assert stage.input_file_type == InputFileType.IR


@pytest.fixture
def ir_pair():
    xml = (_models_dir() / "net.xml").resolve()
    bin_ = (_models_dir() / "net.bin").resolve()
    xml.write_text("<net/>")
    bin_.write_bytes(b"\x00")
    return xml, bin_


def test_extract_bin_xml_from_bin(ir_pair: tuple[Path, Path]):
    xml, bin_ = ir_pair
    got_bin, got_xml = _extract_bin_xml_from_ir(str(bin_))
    assert got_bin == bin_
    assert got_xml == xml


def test_extract_bin_xml_from_xml(ir_pair: tuple[Path, Path]):
    xml, bin_ = ir_pair
    got_bin, got_xml = _extract_bin_xml_from_ir(str(xml))
    assert got_bin == bin_
    assert got_xml == xml


def test_extract_bin_xml_path_object_accepted(ir_pair: tuple[Path, Path]):
    xml, bin_ = ir_pair
    got_bin, got_xml = _extract_bin_xml_from_ir(xml)
    assert (got_bin, got_xml) == (bin_, xml)


def test_extract_bin_xml_bad_suffix_raises_value_error():
    with pytest.raises(ValueError, match=r"does not have \.bin or \.xml"):
        _extract_bin_xml_from_ir("model.onnx")


def test_extract_bin_xml_non_path_raises_type_error():
    with pytest.raises(TypeError, match="must be str or Path"):
        _extract_bin_xml_from_ir(123)


def test_extract_bin_xml_missing_bin_raises_value_error():
    # Only the .xml exists -> resolving the .bin fails.
    xml = (_models_dir() / "lonely.xml").resolve()
    xml.write_text("<net/>")
    with pytest.raises(ValueError, match=r"`bin_path`.*was not found"):
        _extract_bin_xml_from_ir(str(xml))


def test_extract_bin_xml_missing_xml_raises_value_error():
    # Only the .bin exists -> resolving the .xml fails.
    bin_ = (_models_dir() / "orphan.bin").resolve()
    bin_.write_bytes(b"\x00")
    with pytest.raises(ValueError, match=r"`xml_path`.*was not found"):
        _extract_bin_xml_from_ir(str(bin_))


def test_output_default_layout_from_shape():
    out = OutputConfig(name="o", shape=[1, 10])
    assert out.layout == "NC"


def test_output_no_shape_no_layout():
    out = OutputConfig(name="o")
    assert out.shape is None
    assert out.layout is None


def test_output_layout_without_shape_raises():
    with pytest.raises(ValueError, match="cannot be provided without"):
        OutputConfig(name="o", layout="NC")


def test_output_layout_length_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        OutputConfig(name="o", shape=[1, 10], layout="NCD")


def test_output_explicit_layout_uppercased():
    out = OutputConfig(name="o", shape=[1, 10], layout="nc")
    assert out.layout == "NC"


def test_encoding_config_defaults():
    enc = EncodingConfig()
    assert enc.from_ == Encoding.RGB
    assert enc.to == Encoding.BGR


def test_encoding_config_alias():
    enc = EncodingConfig.model_validate({"from": "BGR", "to": "RGB"})
    assert enc.from_ == Encoding.BGR
    assert enc.to == Encoding.RGB


def test_input_encoding_none_defaults_to_rgb_bgr():
    inp = _input_config(name="i", encoding=None)
    assert (inp.encoding.from_, inp.encoding.to) == (
        Encoding.RGB,
        Encoding.BGR,
    )


def test_input_encoding_empty_dict_defaults():
    inp = _input_config(name="i", encoding={})
    assert (inp.encoding.from_, inp.encoding.to) == (
        Encoding.RGB,
        Encoding.BGR,
    )


def test_input_encoding_string_sets_both():
    inp = _input_config(name="i", encoding="RGB")
    assert inp.encoding.from_ == inp.encoding.to == Encoding.RGB


def test_input_encoding_gray_from_forces_gray_gray():
    inp = _input_config(name="i", encoding={"from": "GRAY", "to": "BGR"})
    assert inp.encoding.from_ == inp.encoding.to == Encoding.GRAY


def test_input_encoding_gray_to_forces_gray_gray():
    inp = _input_config(name="i", encoding={"from": "RGB", "to": "GRAY"})
    assert inp.encoding.from_ == inp.encoding.to == Encoding.GRAY


def test_grayscale_input_sets_gray_encoding():
    inp = InputConfig(name="i", shape=[1, 1, 64, 64], layout="NCHW")
    assert inp.encoding.from_ == inp.encoding.to == Encoding.GRAY


def test_non_grayscale_channel_kept():
    inp = InputConfig(name="i", shape=[1, 3, 64, 64], layout="NCHW")
    assert inp.encoding.from_ == Encoding.RGB


def test_multichannel_with_channel_dim_keeps_rgb():
    # "C" present but the channel dim is not 1 -> not grayscale, stays RGB.
    inp = InputConfig(name="i", shape=[1, 10], layout="NC")
    assert inp.encoding.from_ == Encoding.RGB


def test_layout_without_channel_dim_keeps_rgb():
    # No "C" in the layout -> the grayscale check early-returns, RGB stays.
    inp = InputConfig(name="i", shape=[1, 10], layout="NA")
    assert inp.encoding.from_ == Encoding.RGB


def test_dynamic_batch_size_set_to_one():
    inp = InputConfig(name="i", shape=[0, 3, 64, 64], layout="NCHW")
    assert inp.shape == [1, 3, 64, 64]


def test_no_layout_short_circuits_grayscale():
    inp = InputConfig(name="i")
    assert inp.layout is None


def test_named_mean_values():
    inp = _input_config(name="i", mean_values="imagenet")
    assert inp.mean_values == [123.675, 116.28, 103.53]


def test_named_scale_values():
    inp = _input_config(name="i", scale_values="imagenet")
    assert inp.scale_values == [58.395, 57.12, 57.375]


def test_scalar_mean_value_broadcasts_over_channels():
    inp = _input_config(name="i", mean_values=127)
    assert inp.mean_values == [127, 127, 127]


def test_explicit_scale_values_pass_through():
    inp = InputConfig(name="i", scale_values=[1.0, 2.0, 3.0])
    assert inp.scale_values == [1.0, 2.0, 3.0]


def test_unset_mean_values_stay_none():
    inp = InputConfig(name="i", mean_values=None)
    assert inp.mean_values is None


def test_encoding_mismatch_requires_onnx_modification():
    inp = _input_config(name="i", encoding={"from": "RGB", "to": "BGR"})
    assert inp.requires_onnx_input_modification()


def test_reverse_only_ignores_mean_and_scale():
    inp = _input_config(name="i", encoding="RGB", mean_values=[1, 2, 3])
    assert not inp.requires_onnx_input_modification(reverse_only=True)


def test_normalization_requires_onnx_modification():
    inp = _input_config(name="i", encoding="RGB", scale_values=[2, 2, 2])
    assert inp.requires_onnx_input_modification()


def test_plain_input_needs_no_onnx_modification():
    inp = _input_config(name="i", encoding="RGB")
    assert not inp.requires_onnx_input_modification()


def test_raw_and_colour_input_properties():
    raw = _input_config(name="i", encoding="NONE")
    assert raw.is_raw_input
    assert not raw.is_color_input
    assert not raw.encoding_mismatch
    color = _input_config(name="i", encoding="RGB")
    assert color.is_color_input


def test_random_calibration_defaults():
    cal = RandomCalibrationConfig()
    assert cal.max_images == 20
    assert cal.mean == 127.5
    assert cal.data_type == DataType.FLOAT32


def test_image_calibration_local_dir(tmp_path: Path):
    data_dir = tmp_path / "calib"
    data_dir.mkdir()
    cal = ImageCalibrationConfig.model_validate({"path": str(data_dir)})
    assert cal.path == data_dir
    assert cal.resize_method == ResizeMethod.RESIZE


def test_image_calibration_none_path_rejected():
    # The ``None`` short-circuit runs, then the required ``Path`` field
    # rejects the resulting ``None``.
    with pytest.raises(ValidationError):
        ImageCalibrationConfig.model_validate({"path": None})


def test_link_requires_output_or_script():
    with pytest.raises(ValueError, match="Either `output` or `script`"):
        LinkCalibrationConfig(stage="s")


def test_link_with_output():
    cal = LinkCalibrationConfig(stage="s", output="out", script=None)
    assert cal.output == "out"
    assert cal.script is None


def test_link_with_inline_script():
    cal = LinkCalibrationConfig(stage="s", script="print('hi')")
    assert cal.script == "print('hi')"


def test_link_with_script_file():
    script = (_models_dir() / "post.py").resolve()
    script.write_text("def run_script(outputs):\n    return outputs\n")
    cal = LinkCalibrationConfig(stage="s", script=str(script))
    assert cal.script is not None
    assert "run_script" in cal.script


def test_superblob_forces_eight_shaves():
    # The mismatch is reported via loguru, not ``warnings``.
    cfg = RVC2Config(superblob=True, number_of_shaves=4)
    assert cfg.number_of_shaves == 8


def test_no_superblob_keeps_shaves():
    cfg = RVC2Config(superblob=False, number_of_shaves=4)
    assert cfg.number_of_shaves == 4


def test_rvc3_config_defaults():
    cfg = RVC3Config()
    assert cfg.pot_target_device.value == "VPU"


def test_hailo_config_defaults():
    cfg = HailoConfig()
    assert cfg.hw_arch == "hailo8"
    assert cfg.optimization_level == 2


def test_encodings_none():
    assert RVC4Config(encodings=None).encodings is None


def test_platform_config_dispatch():
    cfg = SingleStageConfig.model_construct()
    assert cfg.get_platform_config(Platform.HAILO) is cfg.hailo
    assert cfg.get_platform_config(Platform.RVC2) is cfg.rvc2
    assert cfg.get_platform_config(Platform.RVC3) is cfg.rvc3
    assert cfg.get_platform_config(Platform.RVC4) is cfg.rvc4


def test_is_symmetric_json_serialization():
    encodings = Encodings.model_validate(
        {
            "activation_encodings": {
                "a": [{"bitwidth": 8, "is_symmetric": True}],
                "b": [{"bitwidth": 8}],
            },
            "param_encodings": {},
        },
    )
    dumped = encodings.model_dump(mode="json")
    assert dumped["activation_encodings"]["a"][0]["is_symmetric"] == "True"
    assert dumped["activation_encodings"]["b"][0]["is_symmetric"] is None


def test_encodings_from_json_string():
    payload = json.dumps({"activation_encodings": {}, "param_encodings": {}})
    cfg = _rvc4_config(encodings=payload)
    assert isinstance(cfg.encodings, Encodings)


def test_encodings_from_dict():
    cfg = _rvc4_config(
        encodings={"activation_encodings": {}, "param_encodings": {}}
    )
    assert isinstance(cfg.encodings, Encodings)


def test_encodings_from_path():
    enc_file = MISC_DIR / "e.json"
    enc_file.write_text(
        json.dumps({"activation_encodings": {}, "param_encodings": {}})
    )
    cfg = _rvc4_config(encodings=str(enc_file))
    assert isinstance(cfg.encodings, Encodings)


def test_quantization_overrides_extracted():
    enc_file = MISC_DIR / "qo.json"
    enc_file.write_text(
        json.dumps({"activation_encodings": {}, "param_encodings": {}})
    )
    cfg = _rvc4_config(
        snpe_onnx_to_dlc_args=["--quantization_overrides", str(enc_file)]
    )
    assert cfg.snpe_onnx_to_dlc_args == []
    assert isinstance(cfg.encodings, Encodings)


def test_quantization_overrides_conflict_raises():
    with pytest.raises(ValueError, match="Cannot specify both"):
        _rvc4_config(
            snpe_onnx_to_dlc_args=["--quantization_overrides", "x.json"],
            encodings={"activation_encodings": {}, "param_encodings": {}},
        )


def test_fp16_disables_calibration():
    cfg = RVC4Config(quantization_mode=QuantizationMode.FP16_STD)
    assert cfg.disable_calibration


def test_non_fp16_keeps_calibration():
    cfg = RVC4Config(quantization_mode=QuantizationMode.INT8_STD)
    assert not cfg.disable_calibration


def test_all_disabled_with_every_optimization_off():
    cfg = ONNXOptimizationsConfig(
        fuse_add_mul_to_bn=False,
        fuse_comb_add_mul_to_conv=False,
        fuse_single_add_mul_to_conv=False,
        fuse_split_concat_to_conv=False,
        substitute_sub_with_add=False,
        substitute_div_with_mul=False,
    )
    assert cfg.all_disabled()


def test_all_disabled_false_by_default():
    assert not ONNXOptimizationsConfig().all_disabled()


def test_disable_onnx_simplification_warns_and_disables():
    dummy = _dummy()
    with pytest.warns(DeprecationWarning, match="disable_onnx_simplification"):
        config = Config.get_config(
            None,
            {
                "input_model": str(dummy),
                "disable_onnx_simplification": True,
            },
        )
    assert _single_stage(config).onnx_simplification is False


def test_onnx_simplification_deprecation_conflict_raises():
    dummy = _dummy()
    with pytest.raises(ValueError, match="Cannot specify both"):
        Config.get_config(
            None,
            {
                "input_model": str(dummy),
                "disable_onnx_simplification": True,
                "onnx_simplification": "onnxsim",
            },
        )


def test_disable_onnx_simplification_false_is_a_noop():
    dummy = _dummy()
    config = Config.get_config(
        None,
        {"input_model": str(dummy), "disable_onnx_simplification": False},
    )
    assert _single_stage(config).onnx_simplification == "onnxsim"


@pytest.mark.parametrize("value", ["all", True])
def test_onnx_optimizations_enabling_values(value: str | bool):
    config = Config.get_config(
        None, {"input_model": str(_dummy()), "onnx_optimizations": value}
    )
    assert not _single_stage(config).onnx_optimizations.all_disabled()


@pytest.mark.parametrize("value", ["none", False, None])
def test_onnx_optimizations_disabling_values(value: str | bool | None):
    config = Config.get_config(
        None, {"input_model": str(_dummy()), "onnx_optimizations": value}
    )
    assert _single_stage(config).onnx_optimizations.all_disabled()


def test_onnx_optimizations_default_to_enabled():
    config = Config.get_config(None, {"input_model": str(_dummy())})
    assert not _single_stage(config).onnx_optimizations.all_disabled()


def test_disable_onnx_optimizations_warns_and_disables():
    dummy = _dummy()
    with pytest.warns(DeprecationWarning, match="disable_onnx_optimizations"):
        config = Config.get_config(
            None,
            {
                "input_model": str(dummy),
                "disable_onnx_optimizations": True,
            },
        )
    assert _single_stage(config).onnx_optimizations.all_disabled()


def test_disable_onnx_optimizations_deprecation_conflict_raises():
    dummy = _dummy()
    with (
        pytest.warns(DeprecationWarning, match="disable_onnx_optimizations"),
        pytest.raises(ValueError, match="Cannot specify both"),
    ):
        Config.get_config(
            None,
            {
                "input_model": str(dummy),
                "disable_onnx_optimizations": True,
                "onnx_optimizations": "all",
            },
        )


def test_disable_onnx_optimizations_false_is_a_noop():
    dummy = _dummy()
    config = Config.get_config(
        None,
        {"input_model": str(dummy), "disable_onnx_optimizations": False},
    )
    assert not _single_stage(config).onnx_optimizations.all_disabled()


def test_input_output_names_inferred_from_metadata():
    config = Config.get_config(None, {"input_model": str(_dummy())})
    stage = _single_stage(config)
    assert [i.name for i in stage.inputs] == ["input0", "input1"]
    assert [o.name for o in stage.outputs] == ["output0", "output1"]
    assert stage.inputs[0].shape == [1, 3, 64, 64]
    assert stage.outputs[0].shape == [1, 10]
    assert stage.outputs[0].layout == "NC"


def test_top_level_values_distributed_to_inputs():
    config = Config.get_config(
        None,
        {
            "input_model": str(_dummy()),
            "mean_values": "imagenet",
            "scale_values": "[255, 255, 255]",
            "data_type": "float16",
            "shape": "[1, 3, 32, 32]",
            "layout": "NCHW",
        },
    )
    stage = _single_stage(config)
    for inp in stage.inputs:
        assert inp.mean_values == [123.675, 116.28, 103.53]
        assert inp.scale_values == [255, 255, 255]
        assert inp.data_type == DataType.FLOAT16
        assert inp.shape == [1, 3, 32, 32]
        assert inp.layout == "NCHW"


def test_intermediate_input_shape_looked_up():
    model = intermediate_info_onnx(_models_dir() / "inter.onnx").resolve()
    config = Config.get_config(
        None,
        {"input_model": str(model), "inputs.0.name": "described_node"},
    )
    stage = _single_stage(config)
    assert stage.inputs[0].name == "described_node"
    assert stage.inputs[0].shape == [1, 3, 64, 64]


def test_intermediate_output_shape_looked_up():
    model = intermediate_info_onnx(_models_dir() / "inter2.onnx").resolve()
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "inputs.0.name": "input0",
            "outputs.0.name": "described_node",
        },
    )
    stage = _single_stage(config)
    assert stage.outputs[0].name == "described_node"
    assert stage.outputs[0].shape == [1, 3, 64, 64]


def test_input_without_name_raises():
    with pytest.raises(ValueError, match="Unable to determine name"):
        Config.get_config(
            None,
            {
                "input_model": str(_dummy()),
                "inputs.0.shape": "[1, 3, 8, 8]",
            },
        )


def test_custom_output_with_shape_and_dtype():
    """Output not present in metadata but given a shape/dtype hits the
    ``None, None`` branch (no ONNX lookup)."""
    config = Config.get_config(
        None,
        {
            "input_model": str(_dummy()),
            "outputs.0.name": "custom",
            "outputs.0.shape": "[1, 4]",
            "outputs.0.data_type": "float32",
        },
    )
    stage = _single_stage(config)
    assert stage.outputs[0].name == "custom"
    assert stage.outputs[0].shape == [1, 4]


def test_no_calibration_defaults_to_random():
    config = Config.get_config(None, {"input_model": str(_dummy())})
    stage = _single_stage(config)
    assert isinstance(stage.inputs[0].calibration, RandomCalibrationConfig)


def test_random_top_level():
    config = Config.get_config(
        None, {"input_model": str(_dummy()), "calibration": "random"}
    )
    stage = _single_stage(config)
    assert isinstance(stage.inputs[0].calibration, RandomCalibrationConfig)


def test_top_level_and_input_calibration_merged(tmp_path: Path):
    data_dir = tmp_path / "calib"
    data_dir.mkdir()
    config = Config.get_config(
        None,
        {
            "input_model": str(_dummy()),
            "calibration.path": str(data_dir),
            "inputs.0.name": "input0",
            "inputs.0.calibration.max_images": "7",
            "inputs.1.name": "input1",
        },
    )
    stage = _single_stage(config)
    cal0 = stage.inputs[0].calibration
    assert isinstance(cal0, ImageCalibrationConfig)
    assert cal0.path == data_dir
    assert cal0.max_images == 7
    # input1 inherits only the top-level path.
    cal1 = stage.inputs[1].calibration
    assert isinstance(cal1, ImageCalibrationConfig)
    assert cal1.max_images == -1


def test_disable_calibration_fans_out_to_all_targets():
    # ``rvc4`` is already present in the data, exercising the branch
    # that skips re-creating an existing platform dict.
    config = Config.get_config(
        None,
        {
            "input_model": str(_dummy()),
            "disable_calibration": True,
            "rvc4.optimization_level": "3",
        },
    )
    stage = _single_stage(config)
    assert stage.rvc2.disable_calibration
    assert stage.rvc3.disable_calibration
    assert stage.rvc4.disable_calibration
    assert stage.rvc4.optimization_level == 3
    assert stage.hailo.disable_calibration


def test_default_yolo_input_shape():
    config = Config.get_config(None, {"input_model": str(_pt_model())})
    stage = _single_stage(config)
    assert stage.input_file_type == InputFileType.PYTORCH
    assert stage.inputs[0].name == "images"
    assert stage.inputs[0].shape == [640, 640]


def test_yolo_input_shape_as_list():
    config = Config.get_config(
        None,
        {
            "input_model": str(_pt_model()),
            "yolo_input_shape": "[320, 240]",
        },
    )
    stage = _single_stage(config)
    # stored reversed
    assert stage.inputs[0].shape == [240, 320]


def test_yolo_input_shape_as_string_with_space():
    config = Config.get_config(
        None,
        {
            "input_model": str(_pt_model()),
            "yolo_input_shape": "512 384",
        },
    )
    stage = _single_stage(config)
    assert stage.inputs[0].shape == [384, 512]


def test_yolo_input_shape_as_single_string():
    config = Config.get_config(
        None,
        {
            "input_model": str(_pt_model()),
            "yolo_input_shape": "'256'",
        },
    )
    stage = _single_stage(config)
    assert stage.inputs[0].shape == [256, 256]


def test_node_info_success():
    model = intermediate_info_onnx(_models_dir() / "n.onnx").resolve()
    shape, dtype = _get_onnx_node_info(model, "described_node")
    assert shape == [1, 3, 64, 64]
    assert dtype == DataType.FLOAT32


def test_node_not_found():
    model = intermediate_info_onnx(_models_dir() / "n2.onnx").resolve()
    with pytest.raises(NameError, match="not found"):
        _get_onnx_node_info(model, "nonexistent")


def test_output_value_info_missing():
    model = intermediate_info_onnx(_models_dir() / "n3.onnx").resolve()
    with pytest.raises(ValueError, match="Output value info"):
        _get_onnx_node_info(model, "undescribed_node")


def test_tensor_info_graph_input():
    model = intermediate_info_onnx(_models_dir() / "t.onnx").resolve()
    shape, dtype = _get_onnx_tensor_info(model, "input0")
    assert shape == [1, 3, 64, 64]
    assert dtype == DataType.FLOAT32


def test_tensor_info_graph_output():
    model = intermediate_info_onnx(_models_dir() / "t2.onnx").resolve()
    shape, _ = _get_onnx_tensor_info(model, "output0")
    assert shape == [1, 3, 64, 64]


def test_value_info_tensor():
    model = intermediate_info_onnx(_models_dir() / "t3.onnx").resolve()
    shape, _ = _get_onnx_tensor_info(model, "described_node")
    assert shape == [1, 3, 64, 64]


def test_tensor_without_shape_info():
    model = intermediate_info_onnx(_models_dir() / "t4.onnx").resolve()
    with pytest.raises(ValueError, match="does not have shape"):
        _get_onnx_tensor_info(model, "undescribed_node")


def test_tensor_not_found():
    model = intermediate_info_onnx(_models_dir() / "t5.onnx").resolve()
    with pytest.raises(NameError, match="not found"):
        _get_onnx_tensor_info(model, "nonexistent")


def test_dynamic_shape_raises():
    model = dynamic_batch_onnx(_models_dir() / "dyn.onnx").resolve()
    with pytest.raises(ValueError, match="Dynamic shapes"):
        _get_onnx_tensor_info(model, "input0")


def test_inter_info_found_as_tensor():
    model = intermediate_info_onnx(_models_dir() / "i.onnx").resolve()
    shape, dtype = _get_onnx_inter_info(model, "described_node")
    assert shape == [1, 3, 64, 64]
    assert dtype == DataType.FLOAT32


def test_inter_info_fallback_to_node():
    model = _named_node_onnx(_models_dir() / "i2.onnx").resolve()
    shape, dtype = _get_onnx_inter_info(model, "mynode")
    assert shape == [1, 3, 8, 8]
    assert dtype == DataType.FLOAT32


def test_inter_info_not_found_returns_none():
    model = intermediate_info_onnx(_models_dir() / "i3.onnx").resolve()
    shape, dtype = _get_onnx_inter_info(model, "totally_absent")
    assert shape is None
    assert dtype is None


def test_renames_node_io():
    src = intermediate_info_onnx(_models_dir() / "src.onnx").resolve()
    dst = (_models_dir() / "renamed.onnx").resolve()
    generate_renamed_onnx(src, {"described_node": "renamed"}, dst)
    model = onnx.load(str(dst))
    tensors = {
        t for node in model.graph.node for t in (*node.input, *node.output)
    }
    assert "renamed" in tensors
    assert "described_node" not in tensors
