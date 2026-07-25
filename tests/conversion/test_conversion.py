"""Conversion tests -- "does the model convert successfully" per platform.

These run inside the per-platform Docker image (`modelconverter shell
<platform> --dev -c pytest ...`) where the vendor tools live; the tests
call ``convert`` in-process (``shell`` sets IN_DOCKER, so the launcher is
skipped) so pytest-cov measures the exporter code directly. They assert a
conversion succeeds and produces an output artifact -- not numerical
fidelity.

Every scenario runs on every applicable platform, and each case carries
its platform marker, so ``pytest -m rvc2`` runs *only* the RVC2
conversions, ``-m rvc4`` only the RVC4 ones, etc. Platform-specific
scenarios (e.g. OpenVINO IR ``xml``+``bin`` input, which only RVC2/RVC3
support) are restricted to those platforms.

Assets are the known-good, purpose-built test-bucket archives/configs
(they carry their own calibration, including the multistage model's
linked calibration); a raw zoo multistage archive cannot be converted
with calibration.
"""

from dataclasses import dataclass, field

import pytest

from modelconverter.__main__ import convert
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target

GS = "gs://luxonis-test-bucket/modelconverter"

ALL_PLATFORMS = ("rvc2", "rvc3", "rvc4", "hailo")
# OpenVINO IR (xml+bin) is only supported by the OpenVINO backends.
IR_PLATFORMS = ("rvc2", "rvc3")

# Keep the (otherwise very slow) Hailo compilation cheap for a smoke run.
_HAILO_FAST_OPTS = (
    "hailo.compression_level",
    "0",
    "hailo.optimization_level",
    "0",
    "hailo.disable_compilation",
    "True",
)


@dataclass(frozen=True)
class Scenario:
    id: str
    """Human-readable scenario id (used in the test id / output dir)."""
    path: str
    """Input path/URL: an NN archive (.tar.xz), a config (.yaml) or an IR."""
    to_format: str
    """Output format: ``nn_archive`` or ``native``."""
    platforms: tuple[str, ...] = ALL_PLATFORMS
    """Platforms this scenario applies to."""
    opts: tuple[str, ...] = field(default_factory=tuple)
    """Extra ``key value`` conversion overrides."""
    main_stage: str | None = None
    """Main stage name (required for multistage configs)."""
    xfail: dict[str, str] = field(default_factory=dict)
    """Platforms where this scenario is a known/expected failure -> reason."""


SCENARIOS: list[Scenario] = [
    # --- single-stage (resnet18), every input/output format combination ---
    Scenario("archive-to-archive", f"{GS}/resnet18.tar.xz", "nn_archive"),
    Scenario("archive-to-native", f"{GS}/resnet18.tar.xz", "native"),
    Scenario("config-to-archive", f"{GS}/resnet18.yaml", "nn_archive"),
    Scenario("config-to-native", f"{GS}/resnet18.yaml", "native"),
    # --- multistage (yolov5n-seg + mult) with full linked calibration ---
    # Supported the same on every platform. The config carries the inter-stage
    # linked calibration (stage-2 `coeffs` via a script, `prototypes` from
    # stage-1 `protos_output`) and is self-contained (all assets on S3). A raw
    # multistage NN archive cannot reconstruct the linked calibration -- its
    # non-image stage-2 inputs would fall back to random calibration and fail.
    Scenario(
        "multistage-config-to-native",
        "shared_with_container/configs/multistage.yaml",
        "native",
        main_stage="yolov5n-seg",
        # Known per-platform limitations converting this multistage model:
        #  * RVC3 (POT) quantization does not support multi-input models, and the
        #    postprocessor stage (`mult`) has two inputs (coeffs + prototypes);
        #  * the Hailo SDK's ONNX parser crashes on it (IndexError in its
        #    internal `is_null_reshape`), not something modelconverter controls.
        xfail={
            "rvc3": "RVC3 quantization does not support multi-input models",
            "hailo": "Hailo SDK ONNX parser fails on this multistage model",
        },
    ),
    # --- tflite input (posenet mobilenet v1), exercises .tflite -> native ---
    # The tflite is converted to ONNX internally, then to the native model.
    # Expected to fail on Hailo (its SDK does not handle this model).
    Scenario(
        "tflite-to-native",
        f"{GS}/posenet_mobilenet_v1.tflite",
        "native",
        xfail={"hailo": "posenet tflite model fails to convert on Hailo"},
    ),
    # --- platform-specific: OpenVINO IR (xml+bin) input (RVC2/RVC3 only) ---
    Scenario(
        "ir-to-archive",
        "shared_with_container/configs/resnet18_IR.yaml",
        "nn_archive",
        platforms=IR_PLATFORMS,
    ),
]


def _marks(platform: str, scenario: Scenario) -> list:
    marks = [getattr(pytest.mark, platform)]
    if platform in scenario.xfail:
        marks.append(
            pytest.mark.xfail(
                reason=scenario.xfail[platform],
                strict=True,
                raises=SystemExit,
            )
        )
    return marks


_CASES = [
    pytest.param(
        platform,
        scenario,
        marks=_marks(platform, scenario),
        id=f"{platform}-{scenario.id}",
    )
    for scenario in SCENARIOS
    for platform in scenario.platforms
]


def _assert_produced(output_name: str, to_format: str) -> None:
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    if to_format == "nn_archive":
        produced = list(out_dir.rglob("*.tar.xz")) + list(
            out_dir.rglob("*.tar")
        )
        assert produced, f"no NN archive produced in {out_dir}"
    else:
        assert any(out_dir.iterdir()), f"no output produced in {out_dir}"


@pytest.mark.parametrize(("platform", "scenario"), _CASES)
def test_convert(platform: str, scenario: Scenario):
    output_name = f"_{platform}-{scenario.id}"
    extra = _HAILO_FAST_OPTS if platform == "hailo" else ()
    convert(
        Target(platform),
        *scenario.opts,
        *extra,
        path=scenario.path,
        output_dir=output_name,
        to=scenario.to_format,
        main_stage=scenario.main_stage,
    )
    _assert_produced(output_name, scenario.to_format)
