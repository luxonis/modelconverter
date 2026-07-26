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

**Input freezing** is tested separately (``test_rvc2_input_freezing``): it
is an OpenVINO-MO-only feature (``frozen_value``), so only RVC2 actually
bakes the input out, and the resulting rank-1 input is rejected by other
backends' parsers -- so it does not belong in the shared multi-platform net.

Like the other conversion tests these run inside the platform Docker image
and only assert a successful conversion, not numerical fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from modelconverter.__main__ import convert
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target
from tests.helpers.onnx_factory import build_toy_integration_onnx

_SIZE = 64

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
class Precision:
    name: str
    """Precision id (used in the test id / output dir)."""
    opts: tuple[str, ...] = ()
    """Conversion overrides selecting this precision."""
    xfail: str | None = None
    """Reason if this precision is a known/expected failure for the platform."""


# Per-platform precisions. Each is a distinct conversion path the backend
# supports (or, when `xfail` is set, a known limitation we still assert on).
PRECISIONS: dict[str, list[Precision]] = {
    "rvc2": [
        Precision("blob", ("rvc2.superblob", "False")),
        Precision("superblob", ("rvc2.superblob", "True")),
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
    # produces an unquantized fp16 DLC (and auto-disables calibration).
    "rvc4": [
        Precision("int8", ("rvc4.quantization_mode", "INT8_STANDARD")),
        Precision(
            "int8-acc", ("rvc4.quantization_mode", "INT8_ACCURACY_FOCUSED")
        ),
        Precision(
            "int8-int16-mixed", ("rvc4.quantization_mode", "INT8_INT16_MIXED")
        ),
        Precision(
            "int8-int16-mixed-acc",
            ("rvc4.quantization_mode", "INT8_INT16_MIXED_ACCURACY_FOCUSED"),
        ),
        Precision("fp16", ("rvc4.quantization_mode", "FP16_STANDARD")),
    ],
    "hailo": [
        Precision("quant", _HAILO_FAST_OPTS),
    ],
}


def _write_toy_config(
    config_path: Path, onnx_path: Path, *, with_flag: bool = False
) -> None:
    """Write a config exercising every preprocessing feature on the toy net."""
    inputs = [
        {
            "name": "bgr",
            "shape": [1, 3, _SIZE, _SIZE],
            "data_type": "float32",
            "mean_values": [10, 20, 30],
            "scale_values": [2, 4, 8],
            "encoding": {"from": "BGR", "to": "BGR"},
        },
        {
            "name": "rgb",
            "shape": [1, 3, _SIZE, _SIZE],
            "data_type": "float32",
            "mean_values": [5, 6, 7],
            "scale_values": [3, 2, 1],
            "encoding": {"from": "RGB", "to": "BGR"},
        },
        {
            "name": "gray",
            "shape": [1, 1, _SIZE, _SIZE],
            "data_type": "float32",
            "mean_values": [128],
            "scale_values": [255],
            "encoding": "GRAY",
        },
    ]
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
    platform: str, precision: Precision, toy_config: Path
):
    output_name = f"_toy-{platform}-{precision.name}"
    convert(
        Target(platform),
        *precision.opts,
        path=str(toy_config),
        output_dir=output_name,
        to="native",
    )
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    assert any(out_dir.iterdir()), f"no output produced in {out_dir}"


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
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    assert any(out_dir.iterdir()), f"no output produced in {out_dir}"
