"""Conversion *precision* tests -- "does the converted model still behave
like the original".

For each case we convert a model to a native model file, then run inference
twice on the same image:

  * the **original** ONNX via onnxruntime, with the config's preprocessing
    applied by hand (``ONNXReferenceInferer``) -- the reference;
  * the **converted** model via its vendor inferer (``get_inferer``), which
    is fed the raw image and normalizes internally.

A faithful conversion keeps every output's cosine similarity to the
reference above a per-platform floor. Quantizing backends (rvc3/rvc4)
legitimately diverge more than fp16 (rvc2), hence the per-platform values.

Adding a model is one ``PrecisionCase`` entry in ``CASES`` -- nothing here
is model-specific. Give it a config whose ``input_model`` is an ONNX (the
reference is onnxruntime), and optionally override the per-platform
thresholds or the input image.

Like the conversion tests these run *inside* the platform Docker image
(``modelconverter shell <platform> --dev -c pytest ...``), so each case
carries its platform marker and ``-m rvc2`` runs only the rvc2 ones.

Hailo is intentionally excluded by default: the cheap smoke settings the
conversion tests use (``disable_compilation`` + optimization level 0)
destroy numeric fidelity, and a full-optimization Hailo compile is far too
slow for CI. A Hailo precision check would need a dedicated full-opt run.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.packages.getters import get_inferer
from modelconverter.utils.config import Config, SingleStageConfig
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.general import sanitize_net_name
from modelconverter.utils.types import Target
from tests.helpers.data import TEST_IMAGE
from tests.helpers.onnx_reference import ONNXReferenceInferer
from tests.helpers.precision import cosine_similarity, locate_converted_model

GS = "gs://luxonis-test-bucket/modelconverter"

DEFAULT_PLATFORMS = ("rvc2", "rvc3", "rvc4")

DEFAULT_THRESHOLD: Final[float] = 0.8


@dataclass(frozen=True)
class PrecisionCase:
    id: str
    """Human-readable case id (used in the test id / output dir)."""
    config: str
    """Config (or archive) whose ``input_model`` is the reference ONNX."""
    platforms: tuple[str, ...] = DEFAULT_PLATFORMS
    """Platforms this case runs on."""
    thresholds: dict[str, float] = field(default_factory=dict)
    """Per-platform cosine floor overrides (falls back to DEFAULT_THRESHOLD)."""
    image: Path = TEST_IMAGE
    """Inference input image."""
    main_stage: str | None = None
    """Main stage name (only needed for multistage configs)."""


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


def _main_stage(cfg: Config, case: PrecisionCase) -> SingleStageConfig:
    if case.main_stage is not None:  # pragma: no cover
        return cfg.get_stage_config(case.main_stage)
    return next(iter(cfg.stages.values()))


@pytest.mark.real_model
@pytest.mark.parametrize(("platform", "case"), _PARAMS)
def test_precision(platform: str, case: PrecisionCase):
    target = Target(platform)
    output_name = sanitize_net_name(f"prec_{platform}_{case.id}")

    convert(
        target,
        path=case.config,
        output_dir=output_name,
        to="native",
        main_stage=case.main_stage,
    )

    cfg, _, _ = get_configs(target, case.config, [])
    stage = _main_stage(cfg, case)
    model_path = locate_converted_model(OUTPUTS_DIR / output_name, platform)

    reference = ONNXReferenceInferer.from_stage(stage).infer(case.image)

    dest = OUTPUTS_DIR / f"{output_name}_infer"
    inferer = get_inferer(
        target, str(model_path), case.image.parent, dest, stage
    )
    converted = inferer.infer({stage.inputs[0].name: case.image})

    assert set(reference) == set(converted), (
        f"output names mismatch: {list(reference)} vs {list(converted)}"
    )
    threshold = case.thresholds.get(platform, DEFAULT_THRESHOLD)
    for ref_name, ref in reference.items():
        conv = converted[ref_name]
        cos = cosine_similarity(ref, conv)
        cos = cosine_similarity(ref, conv)
        assert cos >= threshold, (
            f"{platform} {case.id} output {ref_name!r}: cosine {cos:.4f} < {threshold}"
        )
