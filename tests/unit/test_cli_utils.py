"""Host-side unit tests for ``modelconverter.cli.utils``."""

from pathlib import Path

import pytest

import modelconverter.cli.utils as cli_utils
from modelconverter.cli.utils import (
    display_output_path,
    extract_preprocessing,
    get_configs,
    get_output_dir_name,
    init_dirs,
)
from modelconverter.utils import ModelconverterException
from modelconverter.utils.config import Config
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    CONFIGS_DIR,
    CONVERSION_MARKER,
    HOST_OUTPUT_DIR_ENV_VAR,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from modelconverter.utils.types import Encoding, Platform
from tests.helpers.archive_factory import (
    default_archive_config,
    pack_archive,
)
from tests.helpers.onnx_factory import standard_dummy_onnx


def _single_stage_config(dummy_onnx: Path, **overrides) -> Config:
    opts = {"input_model": str(dummy_onnx), "shape": [1, 3, 64, 64]}
    opts.update(overrides)
    return Config.get_config(None, opts)


def test_output_dir_default_name():
    out = get_output_dir_name(Platform.RVC4, "my_model", None)
    assert out.parent == OUTPUTS_DIR
    assert out.name.startswith("my_model_to_rvc4_")


@pytest.mark.parametrize(
    (
        "in_docker_context",
        "host_output_dir",
        "path",
        "expected",
    ),
    [
        (
            True,
            "/home/user/project/output",
            Path("/app/output/model/model.rvc4.tar.xz"),
            "/home/user/project/output/model/model.rvc4.tar.xz",
        ),
        (
            True,
            None,
            Path("/app/output/model/model.rvc4.tar.xz"),
            "/app/output/model/model.rvc4.tar.xz",
        ),
        (
            True,
            "/home/user/project/output",
            Path("/outside-output/model.rvc4.tar.xz"),
            "/outside-output/model.rvc4.tar.xz",
        ),
        (
            False,
            "/home/user/project/output",
            Path("/app/output/model/model.rvc4.tar.xz"),
            "/app/output/model/model.rvc4.tar.xz",
        ),
        (
            True,
            "/home/user/project/output",
            Path("/app/output/../other/model.rvc4.tar.xz"),
            "/app/output/../other/model.rvc4.tar.xz",
        ),
        (
            True,
            r"C:\Users\me\project\output",
            Path("/app/output/model/model.rvc4.tar.xz"),
            r"C:\Users\me\project\output\model\model.rvc4.tar.xz",
        ),
    ],
)
def test_display_output_path(
    monkeypatch: pytest.MonkeyPatch,
    in_docker_context: bool,
    host_output_dir: str | None,
    path: Path,
    expected: str,
):
    if in_docker_context:
        monkeypatch.setenv("IN_DOCKER", "1")
    else:
        monkeypatch.delenv("IN_DOCKER", raising=False)

    if host_output_dir is None:
        monkeypatch.delenv(HOST_OUTPUT_DIR_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(HOST_OUTPUT_DIR_ENV_VAR, host_output_dir)

    monkeypatch.setattr(cli_utils, "OUTPUTS_DIR", Path("/app/output"))

    assert display_output_path(path) == expected


def test_output_dir_explicit_and_previous_conversion_is_wiped():
    existing = OUTPUTS_DIR / "custom"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / CONVERSION_MARKER).touch()
    (existing / "stale.txt").write_text("stale")
    out = get_output_dir_name(Platform.RVC2, "model", "custom")
    assert out == OUTPUTS_DIR / "custom"
    # A directory this tool produced is the one case a rerun may clear.
    assert not (OUTPUTS_DIR / "custom" / "stale.txt").exists()


def test_output_dir_explicit_and_empty_is_wiped():
    (OUTPUTS_DIR / "empty").mkdir(parents=True, exist_ok=True)
    out = get_output_dir_name(Platform.RVC2, "model", "empty")
    assert out == OUTPUTS_DIR / "empty"


def test_output_dir_refuses_a_directory_it_did_not_produce():
    """`OUTPUTS_DIR` is the user's `./output`, so a name that already
    holds something else -- notes, a previous run's inference results --
    must not be deleted just because it was passed to
    `--output-dir`.
    """
    foreign = OUTPUTS_DIR / "notes"
    foreign.mkdir(parents=True, exist_ok=True)
    (foreign / "keep.txt").write_text("mine")

    with pytest.raises(ModelconverterException, match="Refusing to overwrite"):
        get_output_dir_name(Platform.RVC2, "model", "notes")
    assert (foreign / "keep.txt").read_text() == "mine"


def test_output_dir_refuses_a_path_that_is_not_a_directory():
    """A rerun clears the directory a previous conversion produced. A
    file of the same name is not that, and deleting it would take the
    user's data with it.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    taken = OUTPUTS_DIR / "taken"
    taken.write_text("mine")

    with pytest.raises(ModelconverterException, match="it is not a directory"):
        get_output_dir_name(Platform.RVC2, "model", "taken")
    assert taken.read_text() == "mine"


@pytest.mark.parametrize("output_dir", ["/abs/results", "../escape", ""])
def test_output_dir_must_stay_under_outputs_dir_in_docker(
    output_dir: str, monkeypatch: pytest.MonkeyPatch
):
    """Only `./output` is mounted into the container: an absolute or
    upward path would be written somewhere the host never sees, and an
    empty one names `./output` itself.
    """
    monkeypatch.setenv("IN_DOCKER", "1")
    with pytest.raises(
        ModelconverterException, match="Invalid `--output-dir`"
    ):
        get_output_dir_name(Platform.RVC2, "model", output_dir)


def test_output_dir_accepts_any_path_natively(tmp_path: Path):
    """A native run has no mount an absolute path could escape, so the
    user's path is written to as given.
    """
    dest = tmp_path / "elsewhere" / "results"

    assert get_output_dir_name(Platform.RVC2, "model", str(dest)) == dest


def test_output_dir_ignores_a_trailing_separator():
    assert (
        get_output_dir_name(Platform.RVC2, "model", "results/")
        == OUTPUTS_DIR / "results"
    )


def test_output_dir_explicit_and_missing_is_kept():
    # output_dir provided but does not exist -> the rmtree branch is skipped.
    out = get_output_dir_name(Platform.RVC2, "model", "fresh")
    assert out == OUTPUTS_DIR / "fresh"


def test_init_dirs():
    init_dirs()
    for d in (CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, CALIBRATION_DIR):
        assert d.is_dir()


def test_get_configs_single_stage_from_opts(dummy_onnx: Path):
    cfg, archive_cfg, main_stage = get_configs(
        Platform.RVC4,
        None,
        ["input_model", str(dummy_onnx), "shape", "[1,3,64,64]"],
    )
    assert archive_cfg is None
    assert main_stage == next(iter(cfg.stages))


def test_get_configs_opts_as_dict(dummy_onnx: Path):
    cfg, archive_cfg, main_stage = get_configs(
        Platform.RVC4,
        None,
        {"input_model": str(dummy_onnx), "shape": [1, 3, 64, 64]},
    )
    assert archive_cfg is None
    assert main_stage in cfg.stages


def test_get_configs_odd_number_of_opts_raises(dummy_onnx: Path):
    with pytest.raises(ValueError, match="Invalid number of overrides"):
        get_configs(Platform.RVC4, None, ["input_model"])


def test_get_configs_multistage_yolov8_seg_main_stage(dummy_onnx: Path):
    cfg_yaml = (
        "name: multi\n"
        "stages:\n"
        "  yolov8n_seg:\n"
        f"    input_model: {dummy_onnx}\n"
        "    shape: [1, 3, 64, 64]\n"
        "  other:\n"
        f"    input_model: {dummy_onnx}\n"
        "    shape: [1, 3, 64, 64]\n"
    )
    cfg_path = CONFIGS_DIR / "multi.yaml"
    cfg_path.write_text(cfg_yaml)
    _cfg, archive_cfg, main_stage = get_configs(
        Platform.RVC4, str(cfg_path), []
    )
    assert archive_cfg is None
    assert main_stage == "yolov8n_seg"


def test_get_configs_from_nn_archive(work_dir: Path):
    onnx_path = standard_dummy_onnx(work_dir / "dummy_model.onnx")
    archive = pack_archive(
        work_dir / "dummy_model.tar",
        onnx_path,
        default_archive_config(),
    )
    cfg, archive_cfg, main_stage = get_configs(
        Platform.RVC4, str(archive), None
    )
    assert archive_cfg is not None
    assert main_stage in cfg.stages


def test_get_configs_multistage_without_seg_stage(dummy_onnx: Path):
    # >1 stage but no yolov8-seg stage -> the detection loop finishes
    # without a break and main_stage stays None.
    cfg_yaml = (
        "name: multi\n"
        "stages:\n"
        "  backbone:\n"
        f"    input_model: {dummy_onnx}\n"
        "    shape: [1, 3, 64, 64]\n"
        "  head:\n"
        f"    input_model: {dummy_onnx}\n"
        "    shape: [1, 3, 64, 64]\n"
    )
    cfg_path = CONFIGS_DIR / "multi_noseg.yaml"
    cfg_path.write_text(cfg_yaml)
    _cfg, archive_cfg, main_stage = get_configs(
        Platform.RVC4, str(cfg_path), []
    )
    assert archive_cfg is None
    assert main_stage is None


def test_extract_preprocessing_rejects_multistage(dummy_onnx: Path):
    cfg_yaml = (
        "name: multi\n"
        "stages:\n"
        "  a:\n"
        f"    input_model: {dummy_onnx}\n"
        "    shape: [1, 3, 64, 64]\n"
        "  b:\n"
        f"    input_model: {dummy_onnx}\n"
        "    shape: [1, 3, 64, 64]\n"
    )
    cfg_path = CONFIGS_DIR / "multi_extract.yaml"
    cfg_path.write_text(cfg_yaml)
    cfg = Config.get_config(str(cfg_path), {})
    with pytest.raises(ValueError, match="single-stage"):
        extract_preprocessing(cfg)


def test_image_input_rgb_planar(dummy_onnx: Path):
    cfg = _single_stage_config(
        dummy_onnx,
        encoding={"from": "RGB", "to": "RGB"},
        mean_values=[1, 2, 3],
        scale_values=[4, 5, 6],
    )
    _cfg, preprocessing = extract_preprocessing(cfg)
    block = preprocessing["input0"]
    assert block.reverse_channels is True
    assert block.dai_type is not None
    assert block.dai_type.endswith("p")
    stage = next(iter(_cfg.stages.values()))
    assert stage.inputs[0].encoding.from_ == Encoding.NONE


def test_raw_input_with_mean_scale(dummy_onnx: Path):
    cfg = _single_stage_config(
        dummy_onnx,
        encoding="NONE",
        mean_values=[1, 2, 3],
        scale_values=[4, 5, 6],
    )
    _cfg, preprocessing = extract_preprocessing(cfg)
    assert "input0" in preprocessing
    assert preprocessing["input0"].reverse_channels is None


def test_raw_input_without_mean_scale(dummy_onnx: Path):
    cfg = _single_stage_config(dummy_onnx, encoding="NONE")
    _cfg, preprocessing = extract_preprocessing(cfg)
    assert "input0" not in preprocessing


def test_image_input_float16_channel_type(dummy_onnx: Path):
    # FLOAT16 image input -> dai_type gets the F16F16F16 channel suffix.
    cfg = _single_stage_config(
        dummy_onnx,
        encoding={"from": "RGB", "to": "RGB"},
        data_type="float16",
    )
    _cfg, preprocessing = extract_preprocessing(cfg)
    dai_type = preprocessing["input0"].dai_type
    assert dai_type is not None
    assert dai_type.startswith("RGBF16F16F16")
