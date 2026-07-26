"""Host-side unit tests for ``modelconverter.cli.utils``."""

from pathlib import Path
from typing import Any

import pytest

from modelconverter.cli.utils import (
    ModelType,
    extract_preprocessing,
    get_configs,
    get_output_dir_name,
    init_dirs,
    slug_to_id,
)
from modelconverter.utils.config import Config
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    CONFIGS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)
from modelconverter.utils.types import Encoding, Target
from tests.helpers.archive_factory import (
    default_archive_config,
    pack_archive,
)
from tests.helpers.onnx_factory import standard_dummy_onnx


def _single_stage_config(dummy_onnx: Path, **overrides) -> Config:
    opts = {"input_model": str(dummy_onnx), "shape": [1, 3, 64, 64]}
    opts.update(overrides)
    return Config.get_config(None, opts)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".onnx", ModelType.ONNX),
        (".tflite", ModelType.TFLITE),
        (".xml", ModelType.IR),
        (".bin", ModelType.IR),
        (".pt", ModelType.PYTORCH),
        (".pth", ModelType.PYTORCH),
    ],
)
def test_from_suffix(suffix: str, expected: ModelType):
    assert ModelType.from_suffix(suffix) == expected


def test_from_suffix_unsupported():
    with pytest.raises(ValueError, match="Unsupported model format"):
        ModelType.from_suffix(".foo")


def test_default_name():
    out = get_output_dir_name(Target.RVC4, "my_model", None)
    assert out.parent == OUTPUTS_DIR
    assert out.name.startswith("my_model_to_rvc4_")


def test_explicit_and_existing_removed():
    existing = OUTPUTS_DIR / "custom"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "stale.txt").write_text("stale")
    out = get_output_dir_name(Target.RVC2, "model", "custom")
    assert out == OUTPUTS_DIR / "custom"
    # Existing dir was wiped before returning.
    assert not (OUTPUTS_DIR / "custom" / "stale.txt").exists()


def test_explicit_non_existing_kept():
    # output_dir provided but does not exist -> the rmtree branch is skipped.
    out = get_output_dir_name(Target.RVC2, "model", "fresh")
    assert out == OUTPUTS_DIR / "fresh"


def test_init_dirs():
    init_dirs()
    for d in (CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, CALIBRATION_DIR):
        assert d.is_dir()


def test_single_stage_from_opts(dummy_onnx: Path):
    cfg, archive_cfg, main_stage = get_configs(
        Target.RVC4,
        None,
        ["input_model", str(dummy_onnx), "shape", "[1,3,64,64]"],
    )
    assert archive_cfg is None
    assert main_stage == next(iter(cfg.stages))


def test_opts_as_dict(dummy_onnx: Path):
    cfg, archive_cfg, main_stage = get_configs(
        Target.RVC4,
        None,
        {"input_model": str(dummy_onnx), "shape": [1, 3, 64, 64]},
    )
    assert archive_cfg is None
    assert main_stage in cfg.stages


def test_odd_number_of_opts(dummy_onnx: Path):
    with pytest.raises(ValueError, match="Invalid number of overrides"):
        get_configs(Target.RVC4, None, ["input_model"])


def test_multistage_yolov8_seg_main_stage(dummy_onnx: Path):
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
    _cfg, archive_cfg, main_stage = get_configs(Target.RVC4, str(cfg_path), [])
    assert archive_cfg is None
    assert main_stage == "yolov8n_seg"


def test_from_nn_archive(work_dir: Path):
    onnx_path = standard_dummy_onnx(work_dir / "dummy_model.onnx")
    archive = pack_archive(
        work_dir / "dummy_model.tar",
        onnx_path,
        default_archive_config(),
    )
    cfg, archive_cfg, main_stage = get_configs(Target.RVC4, str(archive), None)
    assert archive_cfg is not None
    assert main_stage in cfg.stages


def test_multistage_without_seg_stage(dummy_onnx: Path):
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
    _cfg, archive_cfg, main_stage = get_configs(Target.RVC4, str(cfg_path), [])
    assert archive_cfg is None
    assert main_stage is None


def test_multistage_raises(dummy_onnx: Path):
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
    assert preprocessing["input0"].dai_type.startswith("RGBF16F16F16")


class _FakeRequest:
    """Stand-in for ``modelconverter.cli.utils.Request``."""

    def __init__(self, responses: list):
        self._responses = list(responses)

    def get(self, endpoint: str, params: dict) -> Any:
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def test_found_public(monkeypatch: pytest.MonkeyPatch):
    import modelconverter.cli.utils as cli_utils

    fake = _FakeRequest([[{"id": "abc123"}]])
    monkeypatch.setattr(cli_utils, "Request", fake)
    assert slug_to_id("my-model", "models") == "abc123"


def test_http_error_then_found_private(monkeypatch: pytest.MonkeyPatch):
    from requests.exceptions import HTTPError

    import modelconverter.cli.utils as cli_utils

    # public lookup errors (suppressed) -> private lookup succeeds.
    fake = _FakeRequest([HTTPError("boom"), [{"id": "priv"}]])
    monkeypatch.setattr(cli_utils, "Request", fake)
    assert slug_to_id("my-model", "modelVersions") == "priv"


def test_not_found_raises(monkeypatch: pytest.MonkeyPatch):
    import modelconverter.cli.utils as cli_utils

    fake = _FakeRequest([[], []])
    monkeypatch.setattr(cli_utils, "Request", fake)
    with pytest.raises(ValueError, match="not found"):
        slug_to_id("missing", "models")
