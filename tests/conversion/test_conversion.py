"""Per-platform "does the model convert" tests.

These run inside the platform Docker image (``modelconverter shell <platform>
--dev -c pytest ...``) where the vendor tools live. ``shell`` sets IN_DOCKER, so
``convert`` runs in-process and pytest-cov measures the exporter code directly.
They assert a conversion succeeds and produces an artifact, not fidelity.

Assets are the purpose-built test-bucket archives/configs, which carry their own
calibration -- including the multistage model's linked calibration, which a raw
zoo archive cannot reconstruct.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import pytest

from modelconverter.__main__ import convert
from modelconverter.utils.types import Target
from tests.helpers.conversion import HAILO_FAST_OPTS, assert_produced
from tests.helpers.platforms import ALL_PLATFORMS, platform_marks

GS = "gs://luxonis-test-bucket/modelconverter"

# OpenVINO IR (xml+bin) is only supported by the OpenVINO backends.
IR_PLATFORMS = ("rvc2", "rvc3")
IR_MODELS = f"{GS}/models"


@dataclass(frozen=True)
class Scenario:
    id: str
    # An NN archive (.tar.xz), a config (.yaml) or an OpenVINO IR.
    path: str
    to_format: Literal["native", "nn_archive"]
    platforms: tuple[str, ...] = ALL_PLATFORMS
    opts: tuple[str, ...] = field(default_factory=tuple)
    # Required for multistage configs.
    main_stage: str | None = None
    # Platforms where this scenario is a known failure -> reason.
    xfail: dict[str, str] = field(default_factory=dict)


SCENARIOS: list[Scenario] = [
    # Single-stage (resnet18), every input/output format combination.
    Scenario("archive-to-archive", f"{GS}/resnet18.tar.xz", "nn_archive"),
    Scenario("archive-to-native", f"{GS}/resnet18.tar.xz", "native"),
    Scenario("config-to-archive", f"{GS}/resnet18.yaml", "nn_archive"),
    Scenario("config-to-native", f"{GS}/resnet18.yaml", "native"),
    # Multistage (yolov5n-seg + mult) with full linked calibration: stage-2
    # `coeffs` via a script, `prototypes` from stage-1 `protos_output`. A raw
    # multistage NN archive cannot reconstruct that, so its non-image stage-2
    # inputs would fall back to random calibration and fail -- hence the config.
    Scenario(
        "multistage-config-to-native",
        "shared_with_container/configs/multistage.yaml",
        "native",
        main_stage="yolov5n-seg",
        xfail={
            # The postprocessor stage (`mult`) takes coeffs + prototypes.
            "rvc3": "RVC3 quantization does not support multi-input models",
            # IndexError in the SDK's own `is_null_reshape`.
            "hailo": "Hailo SDK ONNX parser fails on this multistage model",
        },
    ),
    # A .tflite input (posenet mobilenet v1), converted to ONNX internally.
    Scenario(
        "tflite-to-native",
        f"{GS}/posenet_mobilenet_v1.tflite",
        "native",
        xfail={"hailo": "posenet tflite model fails to convert on Hailo"},
    ),
    Scenario(
        "ir-to-archive",
        "shared_with_container/configs/resnet18_IR.yaml",
        "nn_archive",
        platforms=IR_PLATFORMS,
    ),
]


def _scenario_options(platform: str, scenario: Scenario) -> tuple[str, ...]:
    if scenario.id != "ir-to-archive":
        return scenario.opts

    version = os.environ["MODELCONVERTER_TARGET_VERSION"].replace(".", "_")
    model = f"{IR_MODELS}/resnet18_{platform}_{version}.xml"
    return (*scenario.opts, "input_model", model)


_CASES = [
    pytest.param(
        platform,
        scenario,
        marks=platform_marks(platform, scenario.xfail.get(platform)),
        id=f"{platform}-{scenario.id}",
    )
    for scenario in SCENARIOS
    for platform in scenario.platforms
]


@pytest.mark.parametrize(("platform", "scenario"), _CASES)
def test_convert(platform: str, scenario: Scenario):
    output_name = f"_{platform}-{scenario.id}"
    extra = HAILO_FAST_OPTS if platform == "hailo" else ()
    convert(
        Target(platform),
        *_scenario_options(platform, scenario),
        *extra,
        path=scenario.path,
        output_dir=output_name,
        to=scenario.to_format,
        main_stage=scenario.main_stage,
    )
    assert_produced(output_name, scenario.to_format)
