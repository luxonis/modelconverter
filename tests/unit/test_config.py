"""Host-side unit tests for ``modelconverter.utils.config``.

Everything runs without network, cloud, Docker or vendor tooling. Tiny
dummy ONNX models are built on the fly (see
``tests/helpers/onnx_factory``) and the sub-config validators are
exercised both directly (by constructing the pydantic models) and
through the full ``Config.get_config`` pipeline.
"""

import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, checker, helper

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
from modelconverter.utils.metadata import Metadata
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    QuantizationMode,
    ResizeMethod,
    Target,
)
from tests.helpers.onnx_factory import (
    dynamic_batch_onnx,
    intermediate_info_onnx,
    standard_dummy_onnx,
)

# --------------------------------------------------------------------------- #
# Local helpers                                                               #
# --------------------------------------------------------------------------- #


def _models_dir() -> Path:
    """The (cwd-relative) models directory created by
    ``_isolate_cwd``."""
    return Path("shared_with_container") / "models"


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


# --------------------------------------------------------------------------- #
# Config-level validators                                                     #
# --------------------------------------------------------------------------- #


class TestStageValidation:
    def test_flat_config_wrapped_and_renamed(self):
        """A flat single-stage config is wrapped and the
        ``default_stage`` placeholder is renamed to the input-model
        stem."""
        dummy = _dummy()
        config = Config.get_config(None, {"input_model": str(dummy)})
        assert config.name == dummy.stem
        assert set(config.stages) == {dummy.stem}
        assert "default_stage" not in config.stages

    def test_flat_config_with_explicit_name_kept(self):
        """An explicit name suppresses the ``default_stage`` rename."""
        dummy = _dummy()
        config = Config.get_config(
            None, {"input_model": str(dummy), "name": "custom"}
        )
        assert config.name == "custom"
        assert set(config.stages) == {"custom"}

    def test_multistage_extras_fanned_into_each_stage(self):
        """Top-level extras are distributed to stages missing them, and
        the derived name joins the stage keys."""
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

    def test_multistage_explicit_name_preserved(self):
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


class TestGetStageConfig:
    def test_none_with_single_stage(self):
        config = Config.get_config(None, {"input_model": str(_dummy())})
        assert config.get_stage_config(None) is _single_stage(config)

    def test_none_with_multiple_stages_raises(self):
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

    def test_explicit_key(self):
        dummy = _dummy()
        config = Config.get_config(
            None,
            {
                "stages.s1.input_model": str(dummy),
                "stages.s2.input_model": str(dummy),
            },
        )
        assert config.get_stage_config("s2") is config.stages["s2"]


# --------------------------------------------------------------------------- #
# SingleStageConfig._download_input_model                                     #
# --------------------------------------------------------------------------- #


class TestDownloadInputModel:
    def test_missing_input_model_raises(self):
        with pytest.raises(ValueError, match="`input_model` must be provided"):
            Config.get_config(None, {})

    def test_onnx_path_resolved_absolute(self):
        dummy = _dummy()
        config = Config.get_config(None, {"input_model": str(dummy)})
        stage = _single_stage(config)
        assert stage.input_model == dummy
        assert stage.input_file_type == InputFileType.ONNX
        assert stage.input_bin is None

    def test_ir_path_extracts_bin_and_xml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """The IR branch splits the ``.xml``/``.bin`` pair;
        ``get_metadata`` is stubbed so no OpenVINO runtime is needed."""
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


# --------------------------------------------------------------------------- #
# _extract_bin_xml_from_ir                                                    #
# --------------------------------------------------------------------------- #


class TestExtractBinXmlFromIr:
    @pytest.fixture
    def ir_pair(self):
        xml = (_models_dir() / "net.xml").resolve()
        bin_ = (_models_dir() / "net.bin").resolve()
        xml.write_text("<net/>")
        bin_.write_bytes(b"\x00")
        return xml, bin_

    def test_from_bin(self, ir_pair: tuple[Path, Path]):
        xml, bin_ = ir_pair
        got_bin, got_xml = _extract_bin_xml_from_ir(str(bin_))
        assert got_bin == bin_
        assert got_xml == xml

    def test_from_xml(self, ir_pair: tuple[Path, Path]):
        xml, bin_ = ir_pair
        got_bin, got_xml = _extract_bin_xml_from_ir(str(xml))
        assert got_bin == bin_
        assert got_xml == xml

    def test_path_object_accepted(self, ir_pair: tuple[Path, Path]):
        xml, bin_ = ir_pair
        got_bin, got_xml = _extract_bin_xml_from_ir(xml)
        assert (got_bin, got_xml) == (bin_, xml)

    def test_bad_suffix_raises_value_error(self):
        with pytest.raises(ValueError, match=r"does not have \.bin or \.xml"):
            _extract_bin_xml_from_ir("model.onnx")

    def test_non_path_raises_type_error(self):
        with pytest.raises(TypeError, match="must be str or Path"):
            _extract_bin_xml_from_ir(123)

    def test_missing_bin_raises_value_error(self):
        # Only the .xml exists -> resolving the .bin fails.
        xml = (_models_dir() / "lonely.xml").resolve()
        xml.write_text("<net/>")
        with pytest.raises(ValueError, match=r"`bin_path`.*was not found"):
            _extract_bin_xml_from_ir(str(xml))

    def test_missing_xml_raises_value_error(self):
        # Only the .bin exists -> resolving the .xml fails.
        bin_ = (_models_dir() / "orphan.bin").resolve()
        bin_.write_bytes(b"\x00")
        with pytest.raises(ValueError, match=r"`xml_path`.*was not found"):
            _extract_bin_xml_from_ir(str(bin_))


# --------------------------------------------------------------------------- #
# OutputConfig                                                                #
# --------------------------------------------------------------------------- #


class TestOutputConfig:
    def test_default_layout_from_shape(self):
        out = OutputConfig(name="o", shape=[1, 10])
        assert out.layout == "NC"

    def test_no_shape_no_layout(self):
        out = OutputConfig(name="o")
        assert out.shape is None
        assert out.layout is None

    def test_layout_without_shape_raises(self):
        with pytest.raises(ValueError, match="cannot be provided without"):
            OutputConfig(name="o", layout="NC")

    def test_layout_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match"):
            OutputConfig(name="o", shape=[1, 10], layout="NCD")

    def test_explicit_layout_uppercased(self):
        out = OutputConfig(name="o", shape=[1, 10], layout="nc")
        assert out.layout == "NC"


# --------------------------------------------------------------------------- #
# EncodingConfig / InputConfig                                                #
# --------------------------------------------------------------------------- #


class TestEncodingConfig:
    def test_defaults(self):
        enc = EncodingConfig()
        assert enc.from_ == Encoding.RGB
        assert enc.to == Encoding.BGR

    def test_alias(self):
        enc = EncodingConfig(**{"from": "BGR", "to": "RGB"})
        assert enc.from_ == Encoding.BGR
        assert enc.to == Encoding.RGB


class TestInputConfigEncoding:
    def test_none_defaults_to_rgb_bgr(self):
        inp = InputConfig(name="i", encoding=None)
        assert (inp.encoding.from_, inp.encoding.to) == (
            Encoding.RGB,
            Encoding.BGR,
        )

    def test_empty_dict_defaults(self):
        inp = InputConfig(name="i", encoding={})
        assert (inp.encoding.from_, inp.encoding.to) == (
            Encoding.RGB,
            Encoding.BGR,
        )

    def test_string_sets_both(self):
        inp = InputConfig(name="i", encoding="RGB")
        assert inp.encoding.from_ == inp.encoding.to == Encoding.RGB

    def test_gray_from_forces_gray_gray(self):
        inp = InputConfig(name="i", encoding={"from": "GRAY", "to": "BGR"})
        assert inp.encoding.from_ == inp.encoding.to == Encoding.GRAY

    def test_gray_to_forces_gray_gray(self):
        inp = InputConfig(name="i", encoding={"from": "RGB", "to": "GRAY"})
        assert inp.encoding.from_ == inp.encoding.to == Encoding.GRAY


class TestInputConfigShapeValidators:
    def test_grayscale_input_sets_gray_encoding(self):
        inp = InputConfig(name="i", shape=[1, 1, 64, 64], layout="NCHW")
        assert inp.encoding.from_ == inp.encoding.to == Encoding.GRAY

    def test_non_grayscale_channel_kept(self):
        inp = InputConfig(name="i", shape=[1, 3, 64, 64], layout="NCHW")
        assert inp.encoding.from_ == Encoding.RGB

    def test_multichannel_with_channel_dim_keeps_rgb(self):
        # "C" present but the channel dim is not 1 -> not grayscale, stays RGB.
        inp = InputConfig(name="i", shape=[1, 10], layout="NC")
        assert inp.encoding.from_ == Encoding.RGB

    def test_layout_without_channel_dim_keeps_rgb(self):
        # No "C" in the layout -> the grayscale check early-returns, RGB stays.
        inp = InputConfig(name="i", shape=[1, 10], layout="NA")
        assert inp.encoding.from_ == Encoding.RGB

    def test_dynamic_batch_size_set_to_one(self):
        inp = InputConfig(name="i", shape=[0, 3, 64, 64], layout="NCHW")
        assert inp.shape == [1, 3, 64, 64]

    def test_no_layout_short_circuits_grayscale(self):
        inp = InputConfig(name="i")
        assert inp.layout is None


class TestInputConfigValueParsing:
    def test_named_mean_values(self):
        inp = InputConfig(name="i", mean_values="imagenet")
        assert inp.mean_values == [123.675, 116.28, 103.53]

    def test_named_scale_values(self):
        inp = InputConfig(name="i", scale_values="imagenet")
        assert inp.scale_values == [58.395, 57.12, 57.375]

    def test_scalar_broadcast(self):
        inp = InputConfig(name="i", mean_values=127)
        assert inp.mean_values == [127, 127, 127]

    def test_list_passthrough(self):
        inp = InputConfig(name="i", scale_values=[1.0, 2.0, 3.0])
        assert inp.scale_values == [1.0, 2.0, 3.0]

    def test_none_stays_none(self):
        inp = InputConfig(name="i", mean_values=None)
        assert inp.mean_values is None


class TestRequiresOnnxInputModification:
    def test_encoding_mismatch_requires(self):
        inp = InputConfig(name="i", encoding={"from": "RGB", "to": "BGR"})
        assert inp.requires_onnx_input_modification() is True

    def test_reverse_only_ignores_normalization(self):
        inp = InputConfig(name="i", encoding="RGB", mean_values=[1, 2, 3])
        assert inp.requires_onnx_input_modification(reverse_only=True) is False

    def test_normalization_requires(self):
        inp = InputConfig(name="i", encoding="RGB", scale_values=[2, 2, 2])
        assert inp.requires_onnx_input_modification() is True

    def test_no_modification_needed(self):
        inp = InputConfig(name="i", encoding="RGB")
        assert inp.requires_onnx_input_modification() is False

    def test_properties(self):
        raw = InputConfig(name="i", encoding="NONE")
        assert raw.is_raw_input is True
        assert raw.is_color_input is False
        assert raw.encoding_mismatch is False
        color = InputConfig(name="i", encoding="RGB")
        assert color.is_color_input is True


# --------------------------------------------------------------------------- #
# Calibration configs                                                         #
# --------------------------------------------------------------------------- #


class TestCalibrationConfigs:
    def test_random_defaults(self):
        cal = RandomCalibrationConfig()
        assert cal.max_images == 20
        assert cal.mean == 127.5
        assert cal.data_type == DataType.FLOAT32

    def test_image_calibration_local_dir(self, tmp_path: Path):
        data_dir = tmp_path / "calib"
        data_dir.mkdir()
        cal = ImageCalibrationConfig(path=str(data_dir))
        assert cal.path == data_dir
        assert cal.resize_method == ResizeMethod.RESIZE

    def test_image_calibration_none_path_rejected(self):
        # The ``None`` short-circuit runs, then the required ``Path`` field
        # rejects the resulting ``None``.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ImageCalibrationConfig(path=None)

    def test_link_requires_output_or_script(self):
        with pytest.raises(ValueError, match="Either `output` or `script`"):
            LinkCalibrationConfig(stage="s")

    def test_link_with_output(self):
        cal = LinkCalibrationConfig(stage="s", output="out", script=None)
        assert cal.output == "out"
        assert cal.script is None

    def test_link_with_inline_script(self):
        cal = LinkCalibrationConfig(stage="s", script="print('hi')")
        assert cal.script == "print('hi')"

    def test_link_with_script_file(self):
        script = (_models_dir() / "post.py").resolve()
        script.write_text("def run_script(outputs):\n    return outputs\n")
        cal = LinkCalibrationConfig(stage="s", script=str(script))
        assert "run_script" in cal.script


# --------------------------------------------------------------------------- #
# Target sub-configs                                                          #
# --------------------------------------------------------------------------- #


class TestRVC2Config:
    def test_superblob_forces_eight_shaves(self):
        # The mismatch is reported via loguru, not ``warnings``.
        cfg = RVC2Config(superblob=True, number_of_shaves=4)
        assert cfg.number_of_shaves == 8

    def test_no_superblob_keeps_shaves(self):
        cfg = RVC2Config(superblob=False, number_of_shaves=4)
        assert cfg.number_of_shaves == 4


class TestRVC3Config:
    def test_defaults(self):
        cfg = RVC3Config()
        assert cfg.pot_target_device.value == "VPU"


class TestHailoConfig:
    def test_defaults(self):
        cfg = HailoConfig()
        assert cfg.hw_arch == "hailo8"
        assert cfg.optimization_level == 2


class TestRVC4Config:
    def test_encodings_none(self):
        assert RVC4Config(encodings=None).encodings is None

    def test_target_config_dispatch(self):
        cfg = SingleStageConfig.model_construct()
        assert cfg.get_target_config(Target.HAILO) is cfg.hailo
        assert cfg.get_target_config(Target.RVC2) is cfg.rvc2
        assert cfg.get_target_config(Target.RVC3) is cfg.rvc3
        assert cfg.get_target_config(Target.RVC4) is cfg.rvc4

    def test_is_symmetric_json_serialization(self):
        encodings = Encodings(
            activation_encodings={
                "a": [{"bitwidth": 8, "is_symmetric": True}],
                "b": [{"bitwidth": 8}],
            },
            param_encodings={},
        )
        dumped = encodings.model_dump(mode="json")
        assert dumped["activation_encodings"]["a"][0]["is_symmetric"] == "True"
        assert dumped["activation_encodings"]["b"][0]["is_symmetric"] is None

    def test_encodings_from_json_string(self):
        payload = json.dumps(
            {"activation_encodings": {}, "param_encodings": {}}
        )
        cfg = RVC4Config(encodings=payload)
        assert isinstance(cfg.encodings, Encodings)

    def test_encodings_from_dict(self):
        cfg = RVC4Config(
            encodings={"activation_encodings": {}, "param_encodings": {}}
        )
        assert isinstance(cfg.encodings, Encodings)

    def test_encodings_from_path(self):
        enc_file = (
            Path("shared_with_container") / "misc" / "e.json"
        ).resolve()
        enc_file.write_text(
            json.dumps({"activation_encodings": {}, "param_encodings": {}})
        )
        cfg = RVC4Config(encodings=str(enc_file))
        assert isinstance(cfg.encodings, Encodings)

    def test_quantization_overrides_extracted(self):
        enc_file = (
            Path("shared_with_container") / "misc" / "qo.json"
        ).resolve()
        enc_file.write_text(
            json.dumps({"activation_encodings": {}, "param_encodings": {}})
        )
        cfg = RVC4Config(
            snpe_onnx_to_dlc_args=["--quantization_overrides", str(enc_file)]
        )
        assert cfg.snpe_onnx_to_dlc_args == []
        assert isinstance(cfg.encodings, Encodings)

    def test_quantization_overrides_conflict_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            RVC4Config(
                snpe_onnx_to_dlc_args=["--quantization_overrides", "x.json"],
                encodings={"activation_encodings": {}, "param_encodings": {}},
            )

    def test_fp16_disables_calibration(self):
        cfg = RVC4Config(quantization_mode=QuantizationMode.FP16_STD)
        assert cfg.disable_calibration is True

    def test_non_fp16_keeps_calibration(self):
        cfg = RVC4Config(quantization_mode=QuantizationMode.INT8_STD)
        assert cfg.disable_calibration is False


# --------------------------------------------------------------------------- #
# ONNXOptimizationsConfig                                                     #
# --------------------------------------------------------------------------- #


class TestONNXOptimizationsConfig:
    def test_all_disabled_true(self):
        cfg = ONNXOptimizationsConfig(
            fuse_add_mul_to_bn=False,
            fuse_comb_add_mul_to_conv=False,
            fuse_single_add_mul_to_conv=False,
            fuse_split_concat_to_conv=False,
            substitute_sub_with_add=False,
            substitute_div_with_mul=False,
        )
        assert cfg.all_disabled() is True

    def test_all_disabled_false(self):
        assert ONNXOptimizationsConfig().all_disabled() is False


# --------------------------------------------------------------------------- #
# Deprecations / onnx-optimization flags (through the full pipeline)          #
# --------------------------------------------------------------------------- #


class TestOnnxSimplificationDeprecation:
    def test_disable_flag_warns_and_sets_false(self):
        dummy = _dummy()
        with pytest.warns(
            DeprecationWarning, match="disable_onnx_simplification"
        ):
            config = Config.get_config(
                None,
                {
                    "input_model": str(dummy),
                    "disable_onnx_simplification": True,
                },
            )
        assert _single_stage(config).onnx_simplification is False

    def test_conflict_raises(self):
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

    def test_disable_flag_false_is_noop(self):
        dummy = _dummy()
        config = Config.get_config(
            None,
            {"input_model": str(dummy), "disable_onnx_simplification": False},
        )
        assert _single_stage(config).onnx_simplification == "onnxsim"


class TestOnnxOptimizations:
    @pytest.mark.parametrize("value", ["all", True])
    def test_enabled(self, value: str | bool):
        config = Config.get_config(
            None, {"input_model": str(_dummy()), "onnx_optimizations": value}
        )
        assert _single_stage(config).onnx_optimizations.all_disabled() is False

    @pytest.mark.parametrize("value", ["none", False, None])
    def test_disabled(self, value: str | bool | None):
        config = Config.get_config(
            None, {"input_model": str(_dummy()), "onnx_optimizations": value}
        )
        assert _single_stage(config).onnx_optimizations.all_disabled() is True

    def test_absent_uses_default(self):
        config = Config.get_config(None, {"input_model": str(_dummy())})
        assert _single_stage(config).onnx_optimizations.all_disabled() is False


class TestDisableOnnxOptimizationsDeprecation:
    def test_deprecated_flag_warns_and_disables(self):
        dummy = _dummy()
        with pytest.warns(
            DeprecationWarning, match="disable_onnx_optimizations"
        ):
            config = Config.get_config(
                None,
                {
                    "input_model": str(dummy),
                    "disable_onnx_optimizations": True,
                },
            )
        assert _single_stage(config).onnx_optimizations.all_disabled() is True

    def test_conflict_raises(self):
        dummy = _dummy()
        with (
            pytest.warns(
                DeprecationWarning, match="disable_onnx_optimizations"
            ),
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

    def test_flag_false_is_noop(self):
        dummy = _dummy()
        config = Config.get_config(
            None,
            {"input_model": str(dummy), "disable_onnx_optimizations": False},
        )
        assert _single_stage(config).onnx_optimizations.all_disabled() is False


# --------------------------------------------------------------------------- #
# SingleStageConfig._validate_model (through the full pipeline)               #
# --------------------------------------------------------------------------- #


class TestValidateModelInference:
    def test_input_output_names_inferred_from_metadata(self):
        config = Config.get_config(None, {"input_model": str(_dummy())})
        stage = _single_stage(config)
        assert [i.name for i in stage.inputs] == ["input0", "input1"]
        assert [o.name for o in stage.outputs] == ["output0", "output1"]
        assert stage.inputs[0].shape == [1, 3, 64, 64]
        assert stage.outputs[0].shape == [1, 10]
        assert stage.outputs[0].layout == "NC"

    def test_top_level_values_distributed_to_inputs(self):
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

    def test_intermediate_input_shape_looked_up(self):
        model = intermediate_info_onnx(_models_dir() / "inter.onnx").resolve()
        config = Config.get_config(
            None,
            {"input_model": str(model), "inputs.0.name": "described_node"},
        )
        stage = _single_stage(config)
        assert stage.inputs[0].name == "described_node"
        assert stage.inputs[0].shape == [1, 3, 64, 64]

    def test_intermediate_output_shape_looked_up(self):
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

    def test_input_without_name_raises(self):
        with pytest.raises(ValueError, match="Unable to determine name"):
            Config.get_config(
                None,
                {
                    "input_model": str(_dummy()),
                    "inputs.0.shape": "[1, 3, 8, 8]",
                },
            )

    def test_custom_output_with_shape_and_dtype(self):
        """Output not present in metadata but given a shape/dtype hits
        the ``None, None`` branch (no ONNX lookup)."""
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


class TestValidateModelCalibration:
    def test_no_calibration_defaults_to_random(self):
        config = Config.get_config(None, {"input_model": str(_dummy())})
        stage = _single_stage(config)
        assert isinstance(stage.inputs[0].calibration, RandomCalibrationConfig)

    def test_random_top_level(self):
        config = Config.get_config(
            None, {"input_model": str(_dummy()), "calibration": "random"}
        )
        stage = _single_stage(config)
        assert isinstance(stage.inputs[0].calibration, RandomCalibrationConfig)

    def test_top_level_and_input_calibration_merged(self, tmp_path: Path):
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
        assert stage.inputs[1].calibration.max_images == -1

    def test_disable_calibration_fans_out_to_all_targets(self):
        # ``rvc4`` is already present in the data, exercising the branch
        # that skips re-creating an existing target dict.
        config = Config.get_config(
            None,
            {
                "input_model": str(_dummy()),
                "disable_calibration": True,
                "rvc4.optimization_level": "3",
            },
        )
        stage = _single_stage(config)
        assert stage.rvc2.disable_calibration is True
        assert stage.rvc3.disable_calibration is True
        assert stage.rvc4.disable_calibration is True
        assert stage.rvc4.optimization_level == 3
        assert stage.hailo.disable_calibration is True


class TestValidateModelPytorch:
    def _pt_model(self) -> Path:
        pt = (_models_dir() / "yolo.pt").resolve()
        pt.write_bytes(b"not-a-real-checkpoint")
        return pt

    def test_default_yolo_input_shape(self):
        config = Config.get_config(
            None, {"input_model": str(self._pt_model())}
        )
        stage = _single_stage(config)
        assert stage.input_file_type == InputFileType.PYTORCH
        assert stage.inputs[0].name == "images"
        assert stage.inputs[0].shape == [640, 640]

    def test_yolo_input_shape_as_list(self):
        config = Config.get_config(
            None,
            {
                "input_model": str(self._pt_model()),
                "yolo_input_shape": "[320, 240]",
            },
        )
        stage = _single_stage(config)
        # stored reversed
        assert stage.inputs[0].shape == [240, 320]

    def test_yolo_input_shape_as_string_with_space(self):
        config = Config.get_config(
            None,
            {
                "input_model": str(self._pt_model()),
                "yolo_input_shape": "512 384",
            },
        )
        stage = _single_stage(config)
        assert stage.inputs[0].shape == [384, 512]

    def test_yolo_input_shape_as_single_string(self):
        config = Config.get_config(
            None,
            {
                "input_model": str(self._pt_model()),
                "yolo_input_shape": "'256'",
            },
        )
        stage = _single_stage(config)
        assert stage.inputs[0].shape == [256, 256]


# --------------------------------------------------------------------------- #
# ONNX introspection helpers (direct)                                         #
# --------------------------------------------------------------------------- #


class TestOnnxNodeInfo:
    def test_success(self):
        model = intermediate_info_onnx(_models_dir() / "n.onnx").resolve()
        shape, dtype = _get_onnx_node_info(model, "described_node")
        assert shape == [1, 3, 64, 64]
        assert dtype == DataType.FLOAT32

    def test_node_not_found(self):
        model = intermediate_info_onnx(_models_dir() / "n2.onnx").resolve()
        with pytest.raises(NameError, match="not found"):
            _get_onnx_node_info(model, "nonexistent")

    def test_output_value_info_missing(self):
        model = intermediate_info_onnx(_models_dir() / "n3.onnx").resolve()
        with pytest.raises(ValueError, match="Output value info"):
            _get_onnx_node_info(model, "undescribed_node")


class TestOnnxTensorInfo:
    def test_graph_input(self):
        model = intermediate_info_onnx(_models_dir() / "t.onnx").resolve()
        shape, dtype = _get_onnx_tensor_info(model, "input0")
        assert shape == [1, 3, 64, 64]
        assert dtype == DataType.FLOAT32

    def test_graph_output(self):
        model = intermediate_info_onnx(_models_dir() / "t2.onnx").resolve()
        shape, _ = _get_onnx_tensor_info(model, "output0")
        assert shape == [1, 3, 64, 64]

    def test_value_info_tensor(self):
        model = intermediate_info_onnx(_models_dir() / "t3.onnx").resolve()
        shape, _ = _get_onnx_tensor_info(model, "described_node")
        assert shape == [1, 3, 64, 64]

    def test_tensor_without_shape_info(self):
        model = intermediate_info_onnx(_models_dir() / "t4.onnx").resolve()
        with pytest.raises(ValueError, match="does not have shape"):
            _get_onnx_tensor_info(model, "undescribed_node")

    def test_tensor_not_found(self):
        model = intermediate_info_onnx(_models_dir() / "t5.onnx").resolve()
        with pytest.raises(NameError, match="not found"):
            _get_onnx_tensor_info(model, "nonexistent")


class TestStaticOnnxShape:
    def test_dynamic_shape_raises(self):
        model = dynamic_batch_onnx(_models_dir() / "dyn.onnx").resolve()
        with pytest.raises(ValueError, match="Dynamic shapes"):
            _get_onnx_tensor_info(model, "input0")


class TestOnnxInterInfo:
    def test_found_as_tensor(self):
        model = intermediate_info_onnx(_models_dir() / "i.onnx").resolve()
        shape, dtype = _get_onnx_inter_info(model, "described_node")
        assert shape == [1, 3, 64, 64]
        assert dtype == DataType.FLOAT32

    def test_fallback_to_node(self):
        model = _named_node_onnx(_models_dir() / "i2.onnx").resolve()
        shape, dtype = _get_onnx_inter_info(model, "mynode")
        assert shape == [1, 3, 8, 8]
        assert dtype == DataType.FLOAT32

    def test_not_found_returns_none(self):
        model = intermediate_info_onnx(_models_dir() / "i3.onnx").resolve()
        shape, dtype = _get_onnx_inter_info(model, "totally_absent")
        assert shape is None
        assert dtype is None


class TestGenerateRenamedOnnx:
    def test_renames_node_io(self):
        src = intermediate_info_onnx(_models_dir() / "src.onnx").resolve()
        dst = (_models_dir() / "renamed.onnx").resolve()
        generate_renamed_onnx(src, {"described_node": "renamed"}, dst)
        model = onnx.load(str(dst))
        tensors = {
            t for node in model.graph.node for t in (*node.input, *node.output)
        }
        assert "renamed" in tensors
        assert "described_node" not in tensors
