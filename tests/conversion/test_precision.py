"""Conversion precision tests -- "does the converted model still behave like the
original".

Each case converts a model, then runs inference twice on the same image: the
original ONNX via onnxruntime with the config's preprocessing applied by hand
(``ONNXReferenceInferer``), and the converted model via its vendor inferer, which
is fed the raw image and normalizes internally. A faithful conversion keeps every
output's cosine similarity above a per-platform floor -- quantizing backends
(rvc3/rvc4) legitimately diverge more than fp16 (rvc2).

Adding a model is one ``PrecisionCase`` entry; nothing here is model-specific.

Hailo is excluded: the cheap smoke settings the conversion tests use destroy
numeric fidelity, and a full-optimization Hailo compile is far too slow for CI.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.packages.getters import get_inferer
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.general import sanitize_net_name
from modelconverter.utils.types import Target
from tests.helpers.data import TEST_IMAGE
from tests.helpers.onnx_reference import ONNXReferenceInferer
from tests.helpers.precision import cosine_similarity, locate_converted_model
from tests.helpers.target_options import target_options

GS = "gs://luxonis-test-bucket/modelconverter"

DEFAULT_PLATFORMS = ("rvc2", "rvc3", "rvc4")
DEFAULT_THRESHOLD: Final[float] = 0.8


@dataclass(frozen=True)
class PrecisionCase:
    """One model whose converted outputs are compared to the ONNX."""

    id: str
    # A config (or archive) whose `input_model` is the reference ONNX.
    config: str
    platforms: tuple[str, ...] = DEFAULT_PLATFORMS
    # Per-platform cosine floors, falling back to DEFAULT_THRESHOLD.
    thresholds: dict[str, float] = field(default_factory=dict)
    image: Path = TEST_IMAGE


CASES: list[PrecisionCase] = [
    PrecisionCase("resnet18", f"{GS}/resnet18.yaml"),
    PrecisionCase(
        "mnist",
        f"{GS}/mnist.yaml",
        platforms=("rvc2", "rvc3", "rvc4", "hailo"),
    ),
]

_PARAMS = [
    pytest.param(
        platform,
        case,
        marks=getattr(pytest.mark, platform),
        id=f"{platform}-{case.id}",
    )
    for case in CASES
    for platform in case.platforms
]


@pytest.mark.parametrize(("platform", "case"), _PARAMS)
def test_precision(platform: str, case: PrecisionCase):
    target = Target(platform)
    options = target_options(target)
    output_name = sanitize_net_name(f"prec_{platform}_{case.id}")

    convert(
        target,
        *options,
        path=case.config,
        output_dir=output_name,
        to="native",
    )

    cfg, _, _ = get_configs(target, case.config, list(options))
    stage = next(iter(cfg.stages.values()))
    model_path = locate_converted_model(OUTPUTS_DIR / output_name, platform)

    reference = ONNXReferenceInferer.from_stage(stage).infer(case.image)
    inferer = get_inferer(
        target,
        str(model_path),
        case.image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer({stage.inputs[0].name: case.image})

    assert set(reference) == set(converted), (
        f"output names mismatch: {list(reference)} vs {list(converted)}"
    )
    threshold = case.thresholds.get(platform, DEFAULT_THRESHOLD)
    for name, ref in reference.items():
        cos = cosine_similarity(ref, converted[name])
        assert cos >= threshold, (
            f"{platform} {case.id} output {name!r}: "
            f"cosine {cos:.4f} < {threshold}"
        )
