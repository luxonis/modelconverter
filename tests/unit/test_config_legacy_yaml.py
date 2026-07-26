"""Backward-compatibility tests for the legacy flat ``config.yaml``
format (top-level model settings with no ``stages:`` block), plus the
deprecated config keys that older configs still rely on.
"""

from pathlib import Path
from textwrap import dedent

import pytest

from modelconverter.utils.config import Config
from modelconverter.utils.constants import CONFIGS_DIR
from modelconverter.utils.types import Encoding


def _write_cfg(text: str, name: str = "legacy.yaml") -> Path:
    path = CONFIGS_DIR / name
    path.write_text(dedent(text))
    return path


def test_flat_single_stage_wraps(dummy_onnx: Path):
    # The classic flat form (no `stages:`) must still parse and be
    # wrapped into a single stage named after the model stem.
    cfg_path = _write_cfg(
        f"""
        input_model: {dummy_onnx}
        mean_values: imagenet
        scale_values: imagenet
        data_type: float32
        shape: [1, 3, 64, 64]

        encoding:
          from: RGB
          to: BGR

        hailo:
          optimization_level: 3
          compression_level: 3
          batch_size: 4
        """
    )
    cfg = Config.get_config(str(cfg_path), {})
    assert cfg.name == dummy_onnx.stem
    assert list(cfg.stages) == [dummy_onnx.stem]
    stage = cfg.stages[dummy_onnx.stem]
    assert stage.inputs[0].encoding.from_ == Encoding.RGB
    assert stage.inputs[0].encoding.to == Encoding.BGR
    assert stage.hailo.optimization_level == 3


def test_multistage_top_level_extras_propagate(dummy_onnx: Path):
    # Legacy multistage configs put shared settings at the top level
    # and expect them fanned out to every stage.
    cfg_path = _write_cfg(
        f"""
        name: legacy_multi
        data_type: float32
        stages:
          stage_a:
            input_model: {dummy_onnx}
            shape: [1, 3, 64, 64]
          stage_b:
            input_model: {dummy_onnx}
            shape: [1, 3, 64, 64]
        """,
        name="legacy_multi.yaml",
    )
    cfg = Config.get_config(str(cfg_path), {})
    assert set(cfg.stages) == {"stage_a", "stage_b"}


def test_disable_onnx_simplification(dummy_onnx: Path):
    cfg_path = _write_cfg(
        f"""
        input_model: {dummy_onnx}
        shape: [1, 3, 64, 64]
        disable_onnx_simplification: true
        """
    )
    with pytest.warns(DeprecationWarning, match="disable_onnx_simplification"):
        cfg = Config.get_config(str(cfg_path), {})
    stage = next(iter(cfg.stages.values()))
    assert stage.onnx_simplification is False


def test_disable_onnx_simplification_conflict(dummy_onnx: Path):
    cfg_path = _write_cfg(
        f"""
        input_model: {dummy_onnx}
        shape: [1, 3, 64, 64]
        disable_onnx_simplification: true
        onnx_simplification: onnxsim
        """
    )
    with pytest.raises(ValueError, match="Cannot specify both"):
        Config.get_config(str(cfg_path), {})
