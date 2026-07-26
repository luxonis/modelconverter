"""Host-side unit tests for ``modelconverter.utils.nn_archive``.

Everything here runs without network, cloud, Docker or vendor tooling:
tiny dummy ONNX models and hand-crafted NN-archive ``config.json`` files
are built on the fly (see ``tests/helpers``), and absolute paths make
``resolve_path`` short-circuit on ``Path.exists()`` so nothing is ever
downloaded. The autouse ``_isolate_cwd`` fixture re-roots every relative
``shared_with_container/*`` dir into the test's own ``tmp_path``, so the
suite is safe under ``pytest -n auto``.
"""

import shutil
from pathlib import Path
from typing import Any

import pytest
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import (
    Head,
    InputType,
    PreprocessingBlock,
)

from modelconverter.utils.config import (
    Config,
    InputConfig,
    RVC2Config,
    RVC3Config,
    RVC4Config,
)
from modelconverter.utils.constants import MISC_DIR, OUTPUTS_DIR
from modelconverter.utils.metadata import get_metadata
from modelconverter.utils.nn_archive import (
    _default_archive_preprocessing,
    _get_io_dtype,
    archive_from_model,
    find_archive_input,
    generate_archive,
    get_archive_input,
    modelconverter_config_to_nn,
    process_nn_archive,
)
from modelconverter.utils.types import DataType, Target
from tests.helpers.archive_factory import (
    default_archive_config,
    pack_archive,
    write_json,
)
from tests.helpers.onnx_factory import (
    grayscale_onnx,
    single_io_onnx,
    standard_dummy_onnx,
)


def _single_input_archive_config(
    preprocessing: dict[str, Any],
    *,
    shape: list[int] | None = None,
    layout: str = "NCHW",
    input_type: str = "image",
    dtype: str = "float32",
    model_name: str = "model.onnx",
) -> dict[str, Any]:
    """A one-input / one-output NN-archive ``config.json`` dict whose
    single input's ``preprocessing`` block is fully controllable."""
    return {
        "config_version": "1.0",
        "model": {
            "metadata": {
                "name": "dummy_model",
                "path": model_name,
                "precision": "float32",
            },
            "inputs": [
                {
                    "name": "input0",
                    "dtype": dtype,
                    "input_type": input_type,
                    "shape": shape or [1, 3, 64, 64],
                    "layout": layout,
                    "preprocessing": preprocessing,
                }
            ],
            "outputs": [
                {"name": "output0", "dtype": "float32", "shape": [1, 10]}
            ],
            "heads": [],
        },
    }


def _pack_single_input(
    work_dir: Path,
    preprocessing: dict[str, Any],
    *,
    grayscale: bool = False,
    **cfg_kwargs: Any,
) -> Path:
    """Builds a matching dummy ONNX + single-input archive and returns
    the ``.tar`` path."""
    model_path = work_dir / "model.onnx"
    if grayscale:
        grayscale_onnx(model_path)
        cfg_kwargs.setdefault("shape", [1, 1, 64, 64])
    else:
        single_io_onnx(model_path)
    config = _single_input_archive_config(preprocessing, **cfg_kwargs)
    return pack_archive(work_dir / "model.tar", model_path, config)


def _stage_input(config: Config) -> InputConfig:
    return next(iter(config.stages.values())).inputs[0]


def test_plain_tar(work_dir: Path):
    onnx = standard_dummy_onnx(work_dir / "dummy_model.onnx")
    tar = pack_archive(
        work_dir / "dummy_model.tar", onnx, default_archive_config()
    )
    config, archive_cfg, main_stage = process_nn_archive(
        Target.RVC4, tar, None
    )
    assert isinstance(config, Config)
    assert isinstance(archive_cfg, NNArchiveConfig)
    assert main_stage in config.stages
    # Extracted next to MISC_DIR / <stem> (no ".tar" suffix kept).
    assert (MISC_DIR / "dummy_model" / "config.json").exists()


def test_tar_xz(work_dir: Path):
    onnx = standard_dummy_onnx(work_dir / "dummy_model.onnx")
    tar = pack_archive(
        work_dir / "dummy_model.tar.xz",
        onnx,
        default_archive_config(),
        mode="w:xz",
    )
    _config, archive_cfg, _main_stage = process_nn_archive(
        Target.RVC4, tar, None
    )
    assert archive_cfg is not None
    # ".tar.xz" -> stem "dummy_model.tar" -> stripped to "dummy_model".
    assert (MISC_DIR / "dummy_model" / "config.json").exists()


def test_directory_input(work_dir: Path):
    model_dir = work_dir / "unpacked"
    model_dir.mkdir()
    standard_dummy_onnx(model_dir / "dummy_model.onnx")
    write_json(default_archive_config(), model_dir / "config.json")
    _config, archive_cfg, _main_stage = process_nn_archive(
        Target.RVC4, model_dir, None
    )
    assert archive_cfg is not None


def test_unsafe_members_are_skipped(work_dir: Path):
    onnx = standard_dummy_onnx(work_dir / "dummy_model.onnx")
    evil = work_dir / "evil.txt"
    evil.write_text("pwned")
    absolute = work_dir / "abs.txt"
    absolute.write_text("pwned")
    tar = pack_archive(
        work_dir / "dummy_model.tar",
        onnx,
        default_archive_config(),
        extra_files={"../evil.txt": evil, "/abs.txt": absolute},
    )
    _config, archive_cfg, _main_stage = process_nn_archive(
        Target.RVC4, tar, None
    )
    assert archive_cfg is not None
    untar = MISC_DIR / "dummy_model"
    # Traversal members are dropped, safe ones extracted.
    assert not (untar.parent / "evil.txt").exists()
    assert (untar / "config.json").exists()


def test_missing_config_raises(work_dir: Path):
    model_dir = work_dir / "no_config"
    model_dir.mkdir()
    standard_dummy_onnx(model_dir / "dummy_model.onnx")
    with pytest.raises(RuntimeError, match="config not found"):
        process_nn_archive(Target.RVC4, model_dir, None)


def test_unknown_path_type_raises(work_dir: Path):
    bogus = work_dir / "not_an_archive.txt"
    bogus.write_text("definitely not a tar")
    with pytest.raises(RuntimeError, match="Unknown NN Archive path"):
        process_nn_archive(Target.RVC4, bogus, None)


ENCODINGS = {
    "activation_encodings": {
        "input0": [{"bitwidth": 8, "scale": 0.1, "offset": 0}]
    },
    "param_encodings": {},
}


def test_custom_encodings_used_for_rvc4(work_dir: Path):
    onnx = standard_dummy_onnx(work_dir / "dummy_model.onnx")
    enc = write_json(ENCODINGS, work_dir / "encondings.json")
    # NOTE: the source misspells the member as "encondings.json"; we must
    # match that exact name to hit the branch.
    tar = pack_archive(
        work_dir / "dummy_model.tar",
        onnx,
        default_archive_config(),
        extra_files={"encondings.json": enc},
    )
    config, _archive_cfg, main_stage = process_nn_archive(
        Target.RVC4, tar, None
    )
    assert config.stages[main_stage].rvc4.encodings is not None


def test_encodings_ignored_for_non_rvc4(work_dir: Path):
    onnx = standard_dummy_onnx(work_dir / "dummy_model.onnx")
    enc = write_json(ENCODINGS, work_dir / "encondings.json")
    tar = pack_archive(
        work_dir / "dummy_model.tar",
        onnx,
        default_archive_config(),
        extra_files={"encondings.json": enc},
    )
    config, _archive_cfg, main_stage = process_nn_archive(
        Target.RVC2, tar, None
    )
    # The packed encodings file is only consumed on the RVC4 path; for RVC2
    # it is ignored and the archive still parses into a usable stage.
    assert main_stage in config.stages


# Covers the ``dai_type is not None`` decision tree.


def test_rgb_planar_conflicting_reverse_and_interleaved(work_dir: Path):
    tar = _pack_single_input(
        work_dir,
        {
            "dai_type": "RGB888p",
            "reverse_channels": False,  # conflicts with RGB
            "interleaved_to_planar": True,  # conflicts with 'p'
        },
    )
    config, *_ = process_nn_archive(Target.RVC4, tar, None)
    inp = _stage_input(config)
    assert (inp.encoding.from_.value, inp.encoding.to.value) == (
        "RGB",
        "BGR",
    )
    assert inp.layout == "NCHW"
    assert inp.mean_values == [0, 0, 0]
    assert inp.scale_values == [1, 1, 1]


def test_bgr_interleaved_conflicting_reverse_and_interleaved(work_dir: Path):
    tar = _pack_single_input(
        work_dir,
        {
            "dai_type": "BGR888i",
            "reverse_channels": True,  # conflicts with BGR
            "interleaved_to_planar": False,  # conflicts with 'i'
        },
    )
    config, *_ = process_nn_archive(Target.RVC4, tar, None)
    inp = _stage_input(config)
    assert inp.encoding.from_.value == "BGR"
    assert inp.layout == "NHWC"


def test_gray_planar_no_conflicts(work_dir: Path):
    tar = _pack_single_input(
        work_dir,
        {"dai_type": "GRAY8p"},
        grayscale=True,
    )
    config, *_ = process_nn_archive(Target.RVC4, tar, None)
    inp = _stage_input(config)
    assert inp.encoding.from_.value == "GRAY"
    assert inp.layout == "NCHW"
    assert inp.mean_values == [0]
    assert inp.scale_values == [1]


def test_unknown_dai_type_defaults_to_rgb(work_dir: Path):
    tar = _pack_single_input(
        work_dir,
        {"dai_type": "XYZ123"},  # neither RGB/BGR/GRAY, ends in neither i/p
    )
    config, *_ = process_nn_archive(Target.RVC4, tar, None)
    inp = _stage_input(config)
    assert (inp.encoding.from_.value, inp.encoding.to.value) == (
        "RGB",
        "BGR",
    )
    # Layout unchanged since dai_type ends in neither 'i' nor 'p'.
    assert inp.layout == "NCHW"


# Covers the ``dai_type is None`` (legacy flags) decision tree.


def test_reverse_true_interleaved_true(work_dir: Path):
    tar = _pack_single_input(
        work_dir,
        {"reverse_channels": True, "interleaved_to_planar": True},
    )
    config, *_ = process_nn_archive(Target.RVC4, tar, None)
    inp = _stage_input(config)
    assert inp.encoding.from_.value == "RGB"
    assert inp.layout == "NHWC"


def test_reverse_false_interleaved_false(work_dir: Path):
    tar = _pack_single_input(
        work_dir,
        {"reverse_channels": False, "interleaved_to_planar": False},
    )
    config, *_ = process_nn_archive(Target.RVC4, tar, None)
    inp = _stage_input(config)
    assert inp.encoding.from_.value == "BGR"
    assert inp.layout == "NCHW"
    assert inp.mean_values == [0, 0, 0]


def test_no_flags_defaults_to_rgb(work_dir: Path):
    tar = _pack_single_input(work_dir, {})
    config, *_ = process_nn_archive(Target.RVC4, tar, None)
    inp = _stage_input(config)
    assert (inp.encoding.from_.value, inp.encoding.to.value) == (
        "RGB",
        "BGR",
    )


def test_grayscale_from_single_channel(work_dir: Path):
    tar = _pack_single_input(work_dir, {}, grayscale=True)
    config, *_ = process_nn_archive(Target.RVC4, tar, None)
    inp = _stage_input(config)
    assert inp.encoding.from_.value == "GRAY"
    assert inp.mean_values == [0]
    assert inp.scale_values == [1]


def test_explicit_mean_scale_not_overwritten(work_dir: Path):
    tar = _pack_single_input(
        work_dir,
        {"mean": [10, 20, 30], "scale": [2, 3, 4]},
    )
    config, *_ = process_nn_archive(Target.RVC4, tar, None)
    inp = _stage_input(config)
    assert inp.mean_values == [10, 20, 30]
    assert inp.scale_values == [2, 3, 4]


def test_raw_input_keeps_none_encoding(work_dir: Path):
    tar = _pack_single_input(
        work_dir,
        {
            "mean": None,
            "scale": None,
            "reverse_channels": None,
            "interleaved_to_planar": None,
            "dai_type": None,
        },
        input_type="raw",
    )
    config, archive_cfg, _main = process_nn_archive(Target.RVC4, tar, None)
    assert archive_cfg.model.inputs[0].input_type == InputType.RAW
    inp = _stage_input(config)
    # Non-image input -> no mean/scale defaults filled.
    assert inp.mean_values is None
    assert inp.scale_values is None


def test_postprocessor_path_adds_stage(work_dir: Path):
    onnx = standard_dummy_onnx(work_dir / "dummy_model.onnx")
    post = single_io_onnx(work_dir / "post.onnx")
    config_dict = default_archive_config()
    config_dict["model"]["heads"] = [
        # A head without a postprocessor_path is skipped (loop continues).
        {
            "parser": "Classification",
            "metadata": {},
            "outputs": ["output1"],
        },
        {
            "parser": "Classification",
            "metadata": {"postprocessor_path": "post.onnx"},
            "outputs": ["output0"],
        },
    ]
    tar = pack_archive(
        work_dir / "dummy_model.tar",
        onnx,
        config_dict,
        extra_files={"post.onnx": post},
    )
    config, _archive_cfg, main_stage = process_nn_archive(
        Target.RVC4, tar, None
    )
    # Two stages: the main model plus the postprocessor.
    assert len(config.stages) == 2
    assert main_stage in config.stages
    assert "post" in config.stages


def _config_from_overrides(dummy_onnx: Path, **overrides: Any) -> Config:
    opts: dict[str, Any] = {"input_model": str(dummy_onnx)}
    opts.update(overrides)
    return Config.get_config(None, opts)


def _config_to_nn(
    config: Config,
    dummy_onnx: Path,
    *,
    orig: NNArchiveConfig | None = None,
    preprocessing: dict[str, PreprocessingBlock] | None = None,
    main_stage: str | None = None,
    target: Target = Target.RVC4,
) -> NNArchiveConfig:
    """``modelconverter_config_to_nn`` with the fixed test boilerplate (output
    name + model path) filled in, exposing only what tests vary."""
    return modelconverter_config_to_nn(
        config,
        Path("out.onnx"),
        orig,
        preprocessing or {},
        main_stage if main_stage is not None else next(iter(config.stages)),
        dummy_onnx,
        target,
    )


@pytest.mark.parametrize(
    ("target", "overrides", "expected"),
    [
        # RVC2: compress_to_fp16 defaults True -> FLOAT16
        (Target.RVC2, {}, DataType.FLOAT16),
        (
            Target.RVC2,
            {"rvc2.compress_to_fp16": False},
            DataType.FLOAT32,
        ),
        # RVC3 mirrors RVC4 for the disable_calibration cases.
        (
            Target.RVC3,
            {
                "rvc3.compress_to_fp16": True,
                "rvc3.disable_calibration": True,
            },
            DataType.FLOAT16,
        ),
        (
            Target.RVC3,
            {
                "rvc3.compress_to_fp16": False,
                "rvc3.disable_calibration": True,
            },
            DataType.FLOAT32,
        ),
        (Target.RVC3, {}, DataType.INT8),
        # RVC4 via quantization_mode.
        (
            Target.RVC4,
            {"rvc4.quantization_mode": "FP16_STANDARD"},
            DataType.FLOAT16,
        ),
        (
            Target.RVC4,
            {"rvc4.disable_calibration": True},
            DataType.FLOAT32,
        ),
        (Target.RVC4, {}, DataType.INT8),
        (Target.HAILO, {}, DataType.INT8),
    ],
)
def test_precision(
    dummy_onnx: Path,
    target: Target,
    overrides: dict[str, Any],
    expected: DataType,
):
    config = _config_from_overrides(dummy_onnx, **overrides)
    nn = _config_to_nn(config, dummy_onnx, target=target)
    assert nn.model.metadata.precision.value == expected.value


@pytest.mark.parametrize(
    "onnx_to_dlc_args",
    [["--float_bitwidth", "16"], ["--float_bitwidth=16"]],
    ids=["pairwise", "equals"],
)
def test_rvc4_fp16_via_snpe_args(
    dummy_onnx: Path, onnx_to_dlc_args: list[str]
):
    """Both the ``--flag value`` and ``--flag=value`` forms select fp16."""
    config = _config_from_overrides(
        dummy_onnx,
        **{
            "rvc4.quantization_mode": "CUSTOM",
            "rvc4.snpe_onnx_to_dlc_args": onnx_to_dlc_args,
            "rvc4.snpe_dlc_graph_prepare_args": ["--use_float_io"],
            "rvc4.disable_calibration": True,
        },
    )
    nn = _config_to_nn(config, dummy_onnx)
    assert nn.model.metadata.precision.value == DataType.FLOAT16.value


def test_output_layout_fallback_on_incompatible_shape(dummy_onnx: Path):
    # output0 real shape is [1, 10]; declare a 3-D shape so
    # guess_new_layout raises and the code falls back to a default.
    config = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "outputs.0.name": "output0",
            "outputs.0.shape": [1, 10, 10],
            "outputs.0.layout": "NCH",
            "outputs.1.name": "output1",
        },
    )
    nn = _config_to_nn(config, dummy_onnx)
    out0 = next(o for o in nn.model.outputs if o.name == "output0")
    # Fallback default layout for [1, 10].
    assert out0.shape == [1, 10]
    assert len(out0.layout) == 2


def test_output_default_layout_when_shape_has_zero(dummy_onnx: Path):
    # A zero in the declared output shape skips guess_new_layout and uses
    # make_default_layout directly.
    config = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "outputs.0.name": "output0",
            "outputs.1.name": "output1",
            "outputs.1.shape": [1, 0, 5, 5],
            "outputs.1.layout": "NCHW",
        },
    )
    nn = _config_to_nn(config, dummy_onnx)
    out1 = next(o for o in nn.model.outputs if o.name == "output1")
    assert out1.shape == [1, 5, 5, 5]
    assert len(out1.layout) == 4


def test_input_default_layout_when_shape_has_zero(dummy_onnx: Path):
    # A zero in a spatial dim forces the make_default_layout branch.
    config = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "inputs.0.name": "input0",
            "inputs.0.shape": [1, 3, 0, 64],
            "inputs.0.layout": "NCHW",
            "inputs.1.name": "input1",
        },
    )
    nn = _config_to_nn(config, dummy_onnx)
    in0 = next(i for i in nn.model.inputs if i.name == "input0")
    # Uses the true metadata shape with a freshly derived layout.
    assert in0.shape == [1, 3, 64, 64]
    assert in0.layout == "NCHW"


def test_preprocessing_override_applied(dummy_onnx: Path):
    config = _config_from_overrides(dummy_onnx)
    block = PreprocessingBlock(
        mean=[1, 2, 3],
        scale=[4, 5, 6],
        reverse_channels=True,
        interleaved_to_planar=False,
        dai_type="RGB888p",
    )
    nn = _config_to_nn(config, dummy_onnx, preprocessing={"input0": block})
    in0 = next(i for i in nn.model.inputs if i.name == "input0")
    assert in0.preprocessing.dai_type == "RGB888p"


def test_multistage_two_stages_sets_postprocessor(dummy_onnx: Path):
    config = Config.get_config(
        None,
        {
            "name": "multi",
            "stages": {
                "main": {"input_model": str(dummy_onnx)},
                "post": {"input_model": str(dummy_onnx)},
            },
        },
    )
    orig = archive_from_model(dummy_onnx)
    orig.model.heads = [
        Head(parser="Classification", metadata={}, outputs=["output0"])
    ]
    nn = _config_to_nn(config, dummy_onnx, orig=orig, main_stage="main")
    assert nn.model.heads[0].metadata.postprocessor_path == "post.onnx"


def test_multistage_without_head_raises(dummy_onnx: Path):
    config = Config.get_config(
        None,
        {
            "name": "multi",
            "stages": {
                "main": {"input_model": str(dummy_onnx)},
                "post": {"input_model": str(dummy_onnx)},
            },
        },
    )
    with pytest.raises(ValueError, match="1 head"):
        _config_to_nn(config, dummy_onnx, main_stage="main")


def test_more_than_two_stages_raises(dummy_onnx: Path):
    config = Config.get_config(
        None,
        {
            "name": "multi",
            "stages": {
                "main": {"input_model": str(dummy_onnx)},
                "post": {"input_model": str(dummy_onnx)},
                "extra": {"input_model": str(dummy_onnx)},
            },
        },
    )
    with pytest.raises(
        NotImplementedError, match="2-stage models are supported"
    ):
        _config_to_nn(config, dummy_onnx, main_stage="main")


def test_raw_input_default_type_without_orig(dummy_onnx: Path):
    # No orig archive -> the input type is derived from is_raw_input, and
    # the raw default preprocessing block is used.
    config = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "inputs.0.name": "input0",
            "inputs.0.encoding": "NONE",
            "inputs.1.name": "input1",
        },
    )
    nn = _config_to_nn(config, dummy_onnx)
    in0 = next(i for i in nn.model.inputs if i.name == "input0")
    assert in0.input_type == InputType.RAW
    assert in0.preprocessing.dai_type is None


def test_raw_input_preprocessing_preserved_from_orig(dummy_onnx: Path):
    # input0 as raw in the config; orig archive carries a raw input whose
    # preprocessing block must be preserved verbatim.
    config = Config.get_config(
        None,
        {
            "input_model": str(dummy_onnx),
            "inputs.0.name": "input0",
            "inputs.0.encoding": "NONE",
            "inputs.1.name": "input1",
        },
    )
    orig = archive_from_model(dummy_onnx)
    orig.model.inputs[0].input_type = InputType.RAW
    orig.model.inputs[0].preprocessing = PreprocessingBlock(
        mean=[9, 9, 9], scale=None
    )
    nn = _config_to_nn(config, dummy_onnx, orig=orig)
    in0 = next(i for i in nn.model.inputs if i.name == "input0")
    assert in0.input_type == InputType.RAW
    assert in0.preprocessing.mean == [9, 9, 9]


def test_iop_input_and_output(dummy_onnx: Path):
    metadata = get_metadata(dummy_onnx)
    cfg = RVC2Config(
        compile_tool_args=[
            "-iop",
            "input0:FP16,input1:FP16,output0:U8,output1:U8",
        ]
    )
    assert (
        _get_io_dtype(Target.RVC2, "input0", metadata, cfg, mode="input")
        == "float16"
    )
    assert (
        _get_io_dtype(Target.RVC2, "output0", metadata, cfg, mode="output")
        == "uint8"
    )


def test_iop_name_not_found_raises(dummy_onnx: Path):
    # `-iop` present but the requested name is absent -> the loop falls
    # through and the whole mapping string is (invalidly) parsed.
    metadata = get_metadata(dummy_onnx)
    cfg = RVC2Config(compile_tool_args=["-iop", "other:FP16"])
    with pytest.raises(ValueError, match="Unsupported IR data type"):
        _get_io_dtype(Target.RVC2, "input0", metadata, cfg, mode="input")


def test_ip_input_and_default_output(dummy_onnx: Path):
    metadata = get_metadata(dummy_onnx)
    cfg = RVC2Config(compile_tool_args=["-ip", "FP16"])
    assert (
        _get_io_dtype(Target.RVC2, "input0", metadata, cfg, mode="input")
        == "float16"
    )
    # Output falls through to the model's own dtype (float32).
    assert (
        _get_io_dtype(Target.RVC2, "output0", metadata, cfg, mode="output")
        == "float32"
    )


def test_op_output_and_default_input(dummy_onnx: Path):
    metadata = get_metadata(dummy_onnx)
    cfg = RVC2Config(compile_tool_args=["-op", "U8"])
    assert (
        _get_io_dtype(Target.RVC2, "output0", metadata, cfg, mode="output")
        == "uint8"
    )
    assert (
        _get_io_dtype(Target.RVC2, "input0", metadata, cfg, mode="input")
        == "float32"
    )


def test_rvc3_default(dummy_onnx: Path):
    metadata = get_metadata(dummy_onnx)
    cfg = RVC3Config()
    assert (
        _get_io_dtype(Target.RVC3, "input0", metadata, cfg, mode="input")
        == "float32"
    )


def test_rvc4_default(dummy_onnx: Path):
    metadata = get_metadata(dummy_onnx)
    cfg = RVC4Config()
    assert (
        _get_io_dtype(Target.RVC4, "output0", metadata, cfg, mode="output")
        == "float32"
    )


def test_raw():
    inp = InputConfig(
        name="x",
        shape=[1, 3, 64, 64],
        layout="NCHW",
        encoding={"from": "NONE", "to": "NONE"},
    )
    block = _default_archive_preprocessing(inp, "NCHW", input_type="raw")
    assert block == {
        "mean": None,
        "scale": None,
        "reverse_channels": None,
        "interleaved_to_planar": None,
        "dai_type": None,
    }


def test_image_float16_interleaved_with_mean_scale():
    inp = InputConfig(
        name="x",
        shape=[1, 3, 64, 64],
        layout="NHWC",
        data_type="float16",
        encoding={"from": "RGB", "to": "RGB"},
        mean_values=[1, 2, 3],
        scale_values=[4, 5, 6],
    )
    block = _default_archive_preprocessing(inp, "NHWC", input_type="image")
    assert block["dai_type"] == "RGBF16F16F16i"
    assert block["reverse_channels"] is True
    assert block["interleaved_to_planar"] is True
    assert block["mean"] == [0, 0, 0]
    assert block["scale"] == [1, 1, 1]


def test_image_uint8_planar_without_mean_scale():
    inp = InputConfig(
        name="x",
        shape=[1, 3, 64, 64],
        layout="NCHW",
        data_type="float32",
        encoding={"from": "BGR", "to": "BGR"},
    )
    block = _default_archive_preprocessing(inp, "NCHW", input_type="image")
    assert block["dai_type"] == "BGR888p"
    assert block["reverse_channels"] is False
    assert block["interleaved_to_planar"] is False
    assert block["mean"] is None
    assert block["scale"] is None


def test_builds_config_from_onnx(dummy_onnx: Path):
    archive = archive_from_model(dummy_onnx)
    names = {inp.name for inp in archive.model.inputs}
    assert names == {"input0", "input1"}
    assert {o.name for o in archive.model.outputs} == {
        "output0",
        "output1",
    }
    assert archive.model.metadata.name == "dummy_model"


def test_get_archive_input_found(dummy_onnx: Path):
    archive = archive_from_model(dummy_onnx)
    assert get_archive_input(archive, "input1").name == "input1"


def test_get_archive_input_missing_raises(dummy_onnx: Path):
    archive = archive_from_model(dummy_onnx)
    with pytest.raises(ValueError, match="not found"):
        get_archive_input(archive, "nope")


def test_find_archive_input_none_cfg():
    assert find_archive_input(None, "input0") is None


def test_find_archive_input_found_and_missing(dummy_onnx: Path):
    archive = archive_from_model(dummy_onnx)
    assert find_archive_input(archive, "input0").name == "input0"
    assert find_archive_input(archive, "nope") is None


def _prepare_output(dummy_onnx: Path, name: str) -> Path:
    out_dir = OUTPUTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({}, out_dir / "buildinfo.json")
    return out_dir


def test_single_model_roundtrip(dummy_onnx: Path):
    out_dir = _prepare_output(dummy_onnx, "single")
    out_model = out_dir / "output.onnx"
    shutil.copy(dummy_onnx, out_model)
    config = _config_from_overrides(dummy_onnx)
    main_stage = next(iter(config.stages))
    archive = generate_archive(
        Target.RVC4,
        config,
        main_stage,
        [out_model],
        out_dir,
        None,
        {},
        dummy_onnx,
        "myarchive",
    )
    assert archive.exists()
    assert archive.name.endswith(".rvc4.tar.xz")
    # Re-parsing the produced archive must succeed.
    reparsed, archive_cfg, _main = process_nn_archive(
        Target.RVC4, archive, None
    )
    assert archive_cfg is not None
    assert isinstance(reparsed, Config)


def test_multiple_models_use_stage_name(dummy_onnx: Path):
    out_dir = _prepare_output(dummy_onnx, "multi")
    out_a = out_dir / "a.onnx"
    out_b = out_dir / "b.onnx"
    shutil.copy(dummy_onnx, out_a)
    shutil.copy(dummy_onnx, out_b)
    config = _config_from_overrides(dummy_onnx)
    main_stage = next(iter(config.stages))
    archive = generate_archive(
        Target.RVC2,
        config,
        main_stage,
        [out_a, out_b],
        out_dir,
        None,
        {},
        dummy_onnx,
        None,  # archive_name None -> falls back to cfg.name
    )
    assert archive.exists()
    assert archive.name.endswith(".rvc2.tar.xz")
