"""Toy-network integration conversion tests.

One purpose-built tiny ONNX net (``build_toy_integration_onnx``) is converted on
every platform, once per precision that platform supports, so a single small
model covers the whole preprocessing surface: per-input mean/scale, an ``RGB`` ->
``BGR`` reversal alongside a ``BGR`` -> ``BGR`` input that keeps its channels, a
single-channel ``gray`` input, and two outputs of different rank.

For rvc2/rvc4 -- whose preprocessing goes through
``onnx_attach_normalization_to_inputs`` -- the converted outputs are also
compared against an fp32 golden reference. Calibration uses a *constant*
synthetic image (``std=0``) and inference uses the same constant, so the
quantized activation range collapses onto that value and int8 stays
near-lossless. rvc3/hailo bake preprocessing via other paths, so they stay
conversion-only.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.platforms.getters import get_inferer
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Platform
from tests.helpers.conversion import (
    HAILO_FAST_OPTS,
    assert_produced,
    toy_net_image_inputs,
    write_config,
)
from tests.helpers.onnx_factory import build_toy_integration_onnx
from tests.helpers.platform_options import (
    platform_options,
    superblob_skip_reason,
)
from tests.helpers.platforms import platform_marks
from tests.helpers.precision import (
    cosine_similarity,
    golden_reference_outputs,
    locate_converted_model,
)

_SIZE = 64
# Constant pixel value used for BOTH calibration and the correctness input, so
# the quantized activation range collapses onto it and int8 is near-lossless.
_CALIB_VALUE = 100
_THRESHOLD = 0.9


@dataclass(frozen=True)
class Precision:
    name: str
    opts: tuple[str, ...] = ()
    # Reason, if this precision is a known failure for the platform.
    xfail: str | None = None
    # Reason, if the current tool version cannot run this precision at all.
    skip: str | None = None
    # Cosine floor, if this precision also gets the fidelity check. Only
    # rvc2/rvc4 can: their preprocessing goes through the same transform as the
    # golden reference, so the outputs are directly comparable.
    correctness: float | None = None


# These two cases own the RVC2 superblob decision explicitly, so unlike the other
# conversion tests they do not take `platform_options` -- where superblob is too
# slow to compile the dedicated case is skipped, rather than quietly flipped to
# `False`, which would just duplicate `blob`.
PRECISIONS: dict[str, list[Precision]] = {
    "rvc2": [
        Precision("blob", ("rvc2.superblob", "False"), correctness=_THRESHOLD),
        Precision(
            "superblob",
            ("rvc2.superblob", "True"),
            skip=superblob_skip_reason(),
            correctness=_THRESHOLD,
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
        # fp16 keeps the baked preprocessing as explicit fp16 Eltwise ops, which
        # the CPU reference backend (all CI has) cannot run, so it stays
        # conversion-only. int8 folds preprocessing into quantization and runs
        # on CPU, hence the correctness check on those.
        Precision(
            "int16",
            ("rvc4.quantization_mode", "INT16_STANDARD"),
        ),
        Precision("fp16", ("rvc4.quantization_mode", "FP16_STANDARD")),
    ],
    "hailo": [
        # The per-input mean/scale give the net channels of very different
        # magnitudes, and Hailo's fixed-point quantizer rejects the resulting
        # dynamic range ("shift delta > 2, cannot quantize").
        Precision(
            "quant",
            HAILO_FAST_OPTS,
            xfail="Hailo quantizer cannot handle the toy net's activation range",
        ),
    ],
}


def _toy_config(workdir: Path, name: str, *, with_flag: bool = False) -> Path:
    """Build the toy ONNX plus a config exercising every preprocessing feature."""
    onnx_path = build_toy_integration_onnx(
        workdir / f"{name}.onnx", size=_SIZE, with_flag=with_flag
    )
    inputs = toy_net_image_inputs(
        {"mean": _CALIB_VALUE, "std": 0, "max_images": 8}, size=_SIZE
    )
    if with_flag:
        # Raw (non-image) inputs frozen to a constant: RVC2-only, and these two
        # cover both `frozen_value` branches -- a scalar and a list (joined into
        # OpenVINO's `->` syntax).
        inputs += [
            {
                "name": "flag",
                "shape": [1],
                "data_type": "int64",
                "encoding": "NONE",
                "frozen_value": [2],
            },
            {
                "name": "chan_scale",
                "shape": [3],
                "data_type": "float32",
                "encoding": "NONE",
                "frozen_value": [1.0, 2.0, 3.0],
            },
        ]
    return write_config(
        workdir,
        name,
        {
            "input_model": str(onnx_path),
            "inputs": inputs,
            "outputs": [{"name": "output"}, {"name": "pooled"}],
        },
    )


@pytest.fixture(scope="module")
def toy_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("toy_integration")
    return _toy_config(workdir, "toy_integration")


@pytest.fixture(scope="module")
def frozen_toy_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("toy_integration_frozen")
    return _toy_config(workdir, "toy_integration_frozen", with_flag=True)


@pytest.fixture(scope="module")
def constant_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A constant-valued image matching the calibration constant."""
    path = tmp_path_factory.mktemp("toy_const") / "const.png"
    cv2.imwrite(
        str(path), np.full((_SIZE, _SIZE, 3), _CALIB_VALUE, dtype=np.uint8)
    )
    return path


def _assert_correct(
    platform: Platform,
    precision: Precision,
    config_path: Path,
    output_name: str,
    image: Path,
    floor: float,
) -> None:
    cfg, _, _ = get_configs(platform, str(config_path), list(precision.opts))
    stage = next(iter(cfg.stages.values()))
    input_configs = {inp.name: inp for inp in stage.inputs}

    reference = golden_reference_outputs(
        Path(stage.input_model),
        input_configs,
        OUTPUTS_DIR / f"{output_name}_golden",
        float(_CALIB_VALUE),
    )
    model_path = locate_converted_model(
        OUTPUTS_DIR / output_name, platform.value
    )
    inferer = get_inferer(
        platform,
        str(model_path),
        image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer(dict.fromkeys(input_configs, image))

    for name, ref in reference.items():
        cos = cosine_similarity(ref, converted[name])
        assert cos >= floor, (
            f"{platform.value} {precision.name} output {name!r}: "
            f"cosine {cos:.5f} < {floor}"
        )


_CASES = [
    pytest.param(
        platform,
        precision,
        marks=platform_marks(platform, precision.xfail, precision.skip),
        id=f"{platform}-{precision.name}",
    )
    for platform, precisions in PRECISIONS.items()
    for precision in precisions
]


@pytest.mark.parametrize(("platform_name", "precision"), _CASES)
def test_toy_integration(
    platform_name: str,
    precision: Precision,
    toy_config: Path,
    constant_image: Path,
):
    platform = Platform(platform_name)
    output_name = f"_toy-{platform_name}-{precision.name}"
    convert(
        platform,
        *precision.opts,
        path=str(toy_config),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)

    if precision.correctness is not None:
        _assert_correct(
            platform,
            precision,
            toy_config,
            output_name,
            constant_image,
            precision.correctness,
        )


@pytest.mark.rvc2
def test_rvc2_input_freezing(frozen_toy_config: Path):
    """Freezing an input to a constant is an OpenVINO MO feature, so it is
    exercised only on RVC2: the INT ``flag`` input is baked out of the graph.

    The resulting rank-1 inputs are rejected by the other backends' parsers,
    which is why they are not part of the shared multi-platform net.
    """
    output_name = "_toy-rvc2-frozen"
    convert(
        Platform.RVC2,
        *platform_options(Platform.RVC2),
        path=str(frozen_toy_config),
        output_dir=output_name,
        to="native",
    )
    assert_produced(output_name)
