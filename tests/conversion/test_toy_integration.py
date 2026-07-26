"""Toy-network integration conversion tests.

A single purpose-built tiny ONNX network (``build_toy_integration_onnx``)
is converted on every platform, once per precision that platform supports,
so one small model exercises the whole preprocessing/conversion surface:

  * **per-input mean/scale** -- each image input uses different values;
  * **colour-space transformation** -- the ``rgb`` input is reversed to BGR
    (``from: RGB, to: BGR``), the ``bgr`` input is not (``BGR -> BGR``);
  * **grayscale** -- the single-channel ``gray`` input;
  * **multi-output** -- two outputs of different rank.

Each platform is parametrized over the precisions it supports: fp16 blob /
superblob for RVC2; quantized / non-quantized for RVC3; the full RVC4
quantization-mode set (int8 standard / accuracy-focused / int8-int16 mixed
variants / fp16); and quantized for Hailo. Cases carry their platform
marker, so ``-m rvc4`` runs only the RVC4 precisions.

**Correctness**: for rvc2/rvc4 (whose preprocessing goes through
``onnx_attach_normalization_to_inputs``) the test also compares the
converted model's outputs against an fp32 golden reference. Calibration uses
a *constant* synthetic image (``RandomCalibrationConfig`` with ``std=0``) and
inference uses the same constant, so the quantized activation range collapses
onto that value and int8 stays near-lossless. rvc3/hailo bake preprocessing
via other paths, so they stay conversion-only.

**Input freezing** is tested separately (``test_rvc2_input_freezing``): it
is an OpenVINO-MO-only feature (``frozen_value``), so only RVC2 actually
bakes the input out, and the resulting rank-1 input is rejected by other
backends' parsers -- so it does not belong in the shared multi-platform net.

Like the other conversion tests these run inside the platform Docker image
and only assert a successful conversion, not numerical fidelity.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.packages.getters import get_inferer
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target
from tests.helpers.conversion import (
    HAILO_FAST_OPTS,
    assert_produced,
    toy_net_image_inputs,
)
from tests.helpers.onnx_factory import build_toy_integration_onnx
from tests.helpers.precision import (
    cosine_similarity,
    golden_reference_outputs,
    locate_converted_model,
)

_SIZE = 64
# Constant pixel value used for BOTH calibration and the correctness input.
# Calibrating (std=0) and inferring on the same constant makes the quantized
# activation range collapse onto that value, so int8 is near-lossless.
_CALIB_VALUE = 100

_THRESHOLD = 0.9


@dataclass(frozen=True)
class Precision:
    name: str
    """Precision id (used in the test id / output dir)."""
    opts: tuple[str, ...] = ()
    """Conversion overrides selecting this precision."""
    xfail: str | None = None
    """Reason if this precision is a known/expected failure for the platform."""
    correctness: float | None = None
    """If set, also assert output fidelity vs the golden at this cosine floor.

    Only rvc2/rvc4 support this: their preprocessing goes through
    ``onnx_attach_normalization_to_inputs`` (the golden reference), so the
    converted output is directly comparable. rvc3/hailo bake preprocessing
    differently and are left as conversion-only.
    """


# Per-platform precisions. Each is a distinct conversion path the backend
# supports (or, when `xfail` is set, a known limitation we still assert on).
# `correctness` opts a precision into the numeric-fidelity check.
PRECISIONS: dict[str, list[Precision]] = {
    "rvc2": [
        Precision("blob", ("rvc2.superblob", "False"), correctness=_THRESHOLD),
        Precision(
            "superblob", ("rvc2.superblob", "True"), correctness=_THRESHOLD
        ),
    ],
    "rvc3": [
        Precision("non-quant", ("rvc3.disable_calibration", "True")),
        Precision(
            "quant",
            (),
            xfail="RVC3 quantization does not support multi-input models",
        ),
    ],
    # RVC4 (SNPE) supports several quantization modes plus fp16. FP16_STANDARD
    # produces an unquantized fp16 DLC (and auto-disables calibration). With
    # constant calibration, int8 should stay near-lossless (tune floors in CI).
    "rvc4": [
        Precision(
            "int8",
            ("rvc4.quantization_mode", "INT8_STANDARD"),
            correctness=_THRESHOLD,
        ),
        Precision(
            "int8-acc",
            ("rvc4.quantization_mode", "INT8_ACCURACY_FOCUSED"),
            correctness=_THRESHOLD,
        ),
        Precision(
            "int8-int16-mixed",
            ("rvc4.quantization_mode", "INT8_INT16_MIXED"),
            correctness=_THRESHOLD,
        ),
        Precision(
            "int8-int16-mixed-acc",
            ("rvc4.quantization_mode", "INT8_INT16_MIXED_ACCURACY_FOCUSED"),
            correctness=_THRESHOLD,
        ),
        # fp16 keeps the baked preprocessing as explicit fp16 Eltwise ops,
        # which the CPU reference backend (all CI has) cannot run -- so this
        # stays conversion-only. int8 folds preprocessing into quantization
        # and runs on CPU, so those carry the correctness check.
        Precision("fp16", ("rvc4.quantization_mode", "FP16_STANDARD")),
    ],
    "hailo": [
        # The per-input mean/scale give the net channels of very different
        # magnitudes; Hailo's fixed-point quantizer rejects the resulting
        # dynamic range ("shift delta > 2, cannot quantize").
        Precision(
            "quant",
            HAILO_FAST_OPTS,
            xfail="Hailo quantizer cannot handle the toy net's activation range (shift delta)",
        ),
    ],
}


def _write_toy_config(
    config_path: Path, onnx_path: Path, *, with_flag: bool = False
) -> None:
    """Write a config exercising every preprocessing feature on the toy net."""
    # Constant calibration (std=0) so quantization is near-lossless for a
    # constant inference input -- see `_CALIB_VALUE`.
    calib = {"mean": _CALIB_VALUE, "std": 0, "max_images": 8}
    inputs = toy_net_image_inputs(calib, size=_SIZE)
    if with_flag:
        inputs.append(
            {
                "name": "flag",
                "shape": [1],
                "data_type": "int64",
                # Raw (non-image) input; frozen to a constant (RVC2-only).
                "encoding": "NONE",
                "frozen_value": [2],
            }
        )
    config = {
        "input_model": str(onnx_path),
        "inputs": inputs,
        "outputs": [{"name": "output"}, {"name": "pooled"}],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


@pytest.fixture(scope="module")
def toy_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the shared (flag-less) toy ONNX + config; return the config."""
    workdir = tmp_path_factory.mktemp("toy_integration")
    onnx_path = workdir / "toy_integration.onnx"
    build_toy_integration_onnx(onnx_path, size=_SIZE)
    config_path = workdir / "toy_integration.yaml"
    _write_toy_config(config_path, onnx_path)
    return config_path


@pytest.fixture(scope="module")
def frozen_toy_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the toy ONNX *with* the freezable flag input + config."""
    workdir = tmp_path_factory.mktemp("toy_integration_frozen")
    onnx_path = workdir / "toy_integration_frozen.onnx"
    build_toy_integration_onnx(onnx_path, size=_SIZE, with_flag=True)
    config_path = workdir / "toy_integration_frozen.yaml"
    _write_toy_config(config_path, onnx_path, with_flag=True)
    return config_path


@pytest.fixture(scope="module")
def constant_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A constant-valued image matching the calibration constant."""
    path = tmp_path_factory.mktemp("toy_const") / "const.png"
    cv2.imwrite(
        str(path), np.full((_SIZE, _SIZE, 3), _CALIB_VALUE, dtype=np.uint8)
    )
    return path


def _assert_correct(
    target: Target,
    precision: Precision,
    config_path: Path,
    output_name: str,
    image: Path,
    floor: float,
) -> None:
    """Compare the converted model's outputs to the fp32 golden reference."""
    cfg, _, _ = get_configs(target, str(config_path), list(precision.opts))
    stage = next(iter(cfg.stages.values()))
    input_configs = {inp.name: inp for inp in stage.inputs}

    golden_dir = OUTPUTS_DIR / f"{output_name}_golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    reference = golden_reference_outputs(
        Path(stage.input_model), input_configs, golden_dir, float(_CALIB_VALUE)
    )

    model_path = locate_converted_model(
        OUTPUTS_DIR / output_name, target.value
    )
    inferer = get_inferer(
        target,
        str(model_path),
        image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer(dict.fromkeys(input_configs, image))

    for (ref_name, ref), conv in zip(
        reference.items(), converted.values(), strict=True
    ):
        cos = cosine_similarity(ref, conv)
        assert cos >= floor, (
            f"{target.value} {precision.name} output {ref_name!r}: "
            f"cosine {cos:.5f} < {floor}"
        )


def _marks(platform: str, precision: Precision) -> list:
    marks = [getattr(pytest.mark, platform)]
    if precision.xfail is not None:
        marks.append(
            pytest.mark.xfail(
                reason=precision.xfail, strict=True, raises=SystemExit
            )
        )
    return marks


_CASES = [
    pytest.param(
        platform,
        precision,
        marks=_marks(platform, precision),
        id=f"{platform}-{precision.name}",
    )
    for platform, precisions in PRECISIONS.items()
    for precision in precisions
]


@pytest.mark.parametrize(("platform", "precision"), _CASES)
def test_toy_integration(
    platform: str,
    precision: Precision,
    toy_config: Path,
    constant_image: Path,
):
    target = Target(platform)
    output_name = f"_toy-{platform}-{precision.name}"
    convert(
        target,
        *precision.opts,
        path=str(toy_config),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)

    # For backends with a comparable golden (rvc2/rvc4), also assert the
    # converted model reproduces the reference outputs.
    if precision.correctness is not None:
        _assert_correct(
            target,
            precision,
            toy_config,
            output_name,
            constant_image,
            precision.correctness,
        )


@pytest.mark.rvc2
def test_rvc2_input_freezing(frozen_toy_config: Path):
    """Freezing an input to a constant (`frozen_value`) is an OpenVINO MO
    feature, so it is exercised only on RVC2: the INT `flag` input is baked
    out of the converted graph."""
    output_name = "_toy-rvc2-frozen"
    convert(
        Target.RVC2,
        path=str(frozen_toy_config),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)
