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
from modelconverter.utils.types import Platform
from tests.helpers.conversion import HAILO_FAST_OPTS, assert_produced
from tests.helpers.platform_options import platform_options
from tests.helpers.platforms import ALL_PLATFORMS, platform_marks

GS = "gs://luxonis-test-bucket/modelconverter"

# OpenVINO IR (xml+bin) is only supported by the OpenVINO backends.
IR_PLATFORMS = ("rvc2", "rvc3")
IR_MODELS = f"{GS}/models"


@dataclass(frozen=True)
class Scenario:
    """One end-to-end conversion the suite runs."""

    id: str
    # An NN archive (.tar.xz), a config (.yaml) or an OpenVINO IR.
    path: str
    to_format: Literal["native", "nn_archive"]
    platforms: tuple[str, ...] = ALL_PLATFORMS
    opts: tuple[str, ...] = field(default_factory=tuple)
    # Required for multistage configs.
    main_stage: str | None = None
    # Store preprocessing in the produced archive instead of the native model.
    archive_preprocess: bool = False
    # Platforms where this scenario is a known failure -> reason.
    xfail: dict[str, str] = field(default_factory=dict)


SCENARIOS: list[Scenario] = [
    # Single-stage (resnet18).
    Scenario("archive-to-archive", f"{GS}/resnet18.tar.xz", "nn_archive"),
    # On rvc2/rvc4, `test_model_evaluation` converts HubAI NN Archives to
    # native models.
    Scenario(
        "archive-to-native",
        f"{GS}/resnet18.tar.xz",
        "native",
        platforms=("rvc3", "hailo"),
    ),
    Scenario(
        "config-to-archive",
        f"{GS}/resnet18.yaml",
        "nn_archive",
        archive_preprocess=True,
    ),
    # On rvc2/rvc3/rvc4, `test_precision[resnet18]` runs this exact conversion
    # and also checks the outputs.
    Scenario(
        "config-to-native",
        f"{GS}/resnet18.yaml",
        "native",
        platforms=("hailo",),
    ),
    # Multistage (yolov5n-seg + mult) with full linked calibration: stage-2
    # `coeffs` via a script, `prototypes` from stage-1 `protos_output`. A raw
    # multistage NN archive cannot reconstruct that, so its non-image stage-2
    # inputs would fall back to random calibration and fail -- hence the config.
    #
    # No RVC3 case: RVC3 cannot quantize the multi-input `mult` stage, and it
    # fails on that only after it has converted the whole first stage. The toy
    # tests reach the same check in seconds.
    Scenario(
        "multistage-config-to-archive",
        "configs/multistage.yaml",
        "nn_archive",
        # Hailo fails on this model before the archive step, see the native
        # case below.
        platforms=("rvc2", "rvc4"),
        main_stage="yolov5n-seg",
    ),
    Scenario(
        "multistage-config-to-native",
        "configs/multistage.yaml",
        "native",
        platforms=("rvc2", "rvc4", "hailo"),
        main_stage="yolov5n-seg",
        xfail={
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
        "configs/resnet18_IR.yaml",
        "nn_archive",
        platforms=IR_PLATFORMS,
    ),
]


def _scenario_options(
    platform_name: str, scenario: Scenario
) -> tuple[str, ...]:
    if scenario.id == "tflite-to-native" and platform_name == "rvc4":
        # RVC4 cannot bake preprocessing into a TFLite model. The source
        # model already expects RGB, so make that no-op contract explicit.
        return (*scenario.opts, "encoding", "RGB")

    if scenario.id != "ir-to-archive":
        return scenario.opts

    version = os.environ["MODELCONVERTER_TOOL_VERSION"].replace(".", "_")
    model = f"{IR_MODELS}/resnet18_{platform_name}_{version}.xml"
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


@pytest.mark.parametrize(("platform_name", "scenario"), _CASES)
def test_convert(platform_name: str, scenario: Scenario):
    platform = Platform(platform_name)
    output_name = f"_{platform_name}-{scenario.id}"
    extra = HAILO_FAST_OPTS if platform_name == "hailo" else ()
    convert(
        platform,
        *_scenario_options(platform_name, scenario),
        # An NN Archive is the default output, so the archive cases also keep
        # the RVC2 default superblob.
        *platform_options(
            platform, superblob=scenario.to_format == "nn_archive"
        ),
        *extra,
        path=scenario.path,
        output_dir=output_name,
        to=scenario.to_format,
        main_stage=scenario.main_stage,
        archive_preprocess=scenario.archive_preprocess,
    )
    assert_produced(output_name, scenario.to_format)
