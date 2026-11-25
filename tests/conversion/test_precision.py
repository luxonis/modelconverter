"""Compare converted models with independently preprocessed ONNX references.

Hailo uses MNIST and the quantized HAR: full ResNet optimization is too
slow for CI, and HEF compilation is unnecessary for CPU inference.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.platforms.getters import get_inferer
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.general import sanitize_net_name
from modelconverter.utils.types import Platform
from tests.helpers.conversion import HAILO_NO_COMPILE_OPTS
from tests.helpers.data import TEST_IMAGE
from tests.helpers.onnx_reference import ONNXReferenceInferer
from tests.helpers.platform_options import platform_options
from tests.helpers.precision import (
    RecordProperty,
    cosine_similarity,
    locate_converted_model,
    relative_error,
)

GS = "gs://luxonis-test-bucket/modelconverter"

DEFAULT_PLATFORMS = ("rvc2", "rvc3", "rvc4")
# Below the lowest cosine that CI measures on each platform for the toy and
# MNIST models: rvc2 0.99993, rvc4 0.99908, Hailo 0.99856, and RVC3 0.99397.
COSINE_FLOORS: Final[dict[str, float]] = {
    "rvc2": 0.999,
    "rvc3": 0.98,
    "rvc4": 0.995,
    "hailo": 0.99,
}


@dataclass(frozen=True)
class PrecisionCase:
    """One model whose converted outputs are compared to the ONNX."""

    id: str
    # A config (or archive) whose `input_model` is the reference ONNX.
    config: str
    platforms: tuple[str, ...] = DEFAULT_PLATFORMS
    calibration_images: dict[str, int] = field(default_factory=dict)
    image: Path = TEST_IMAGE


CASES: list[PrecisionCase] = [
    # The shared config limits calibration to 20 images for conversion smoke
    # tests. ResNet precision needs POT's full 300-image statistics subset.
    PrecisionCase(
        "resnet18",
        f"{GS}/resnet18.yaml",
        calibration_images={"rvc3": 300},
    ),
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


@pytest.mark.parametrize(("platform_name", "case"), _PARAMS)
def test_precision(
    platform_name: str, case: PrecisionCase, record_property: RecordProperty
):
    platform = Platform(platform_name)
    options = platform_options(platform)
    if count := case.calibration_images.get(platform_name):
        options = (*options, "calibration.max_images", str(count))
        record_property(f"{platform_name} {case.id} calibration images", count)
    if platform is Platform.HAILO:
        options = (*options, *HAILO_NO_COMPILE_OPTS)
    output_name = sanitize_net_name(f"prec_{platform_name}_{case.id}")

    convert(
        platform,
        *options,
        path=case.config,
        output_dir=output_name,
        to="native",
    )

    cfg, _, _ = get_configs(platform, case.config, list(options))
    stage = next(iter(cfg.stages.values()))
    model_path = locate_converted_model(
        OUTPUTS_DIR / output_name, platform_name
    )

    reference = ONNXReferenceInferer.from_stage(stage).infer(case.image)
    inferer = get_inferer(
        platform,
        str(model_path),
        case.image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer({stage.inputs[0].name: case.image})

    assert set(reference) == set(converted), (
        f"output names mismatch: {list(reference)} vs {list(converted)}"
    )
    threshold = COSINE_FLOORS[platform_name]
    for name, ref in reference.items():
        actual = converted[name]
        if (
            platform is Platform.HAILO
            and ref.ndim == 2
            and ref.shape[0] == 1
            and actual.shape in {(ref.shape[1],), (1, 1, ref.shape[1])}
        ):
            actual = actual.reshape(ref.shape)
        cos = cosine_similarity(ref, actual)
        label = f"{platform_name} {case.id} {name}"
        record_property(f"{label} cosine", round(cos, 5))
        record_property(
            f"{label} error", round(relative_error(ref, actual), 5)
        )
        assert cos >= threshold, (
            f"{platform_name} {case.id} output {name!r}: "
            f"cosine {cos:.4f} < {threshold}"
        )
