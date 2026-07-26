"""Toy multistage conversion + correctness tests.

Builds a small, fully-controllable multistage network out of the toy pieces:

    first  (toy net)  ->\
                         > third (aggregator: from_first * from_second + bias)
    second (toy net)  ->/

``third``'s two inputs are produced by *linked calibration* from the two
upstream stages, exercising BOTH linking options modelconverter supports:

  * ``from_first``  <- ``{stage: first, output: output}``  (direct output link)
  * ``from_second`` <- ``{stage: second, script: ...}``    (script transform)

The script combines ``second``'s two outputs (``output + pooled``). Stage
order in the config is topological (first, second, third) so the upstream
stages are converted before ``third``'s linked calibration runs their
inferers.

Tests run inside the platform Docker image, e.g.::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k toy_multistage'

Conversion is checked on every platform (default tool-version); correctness
(vs an fp32 golden pipeline) is checked on rvc2/rvc4, matching the
single-stage toy (their preprocessing goes through the golden path and their
DLC/IR runs on the CI CPU backend).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest
import yaml

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.packages.getters import get_inferer
from modelconverter.utils.config import Config
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target
from tests.helpers.onnx_factory import (
    build_toy_aggregator_onnx,
    build_toy_integration_onnx,
)
from tests.helpers.precision import (
    cosine_similarity,
    golden_reference_outputs,
    locate_converted_model,
)

_SIZE = 64
_CALIB_VALUE = 100
_THRESHOLD = 0.9

ALL_PLATFORMS = ("rvc2", "rvc3", "rvc4", "hailo")
CORRECTNESS_PLATFORMS = ("rvc2", "rvc4")
MAIN_STAGE = "third"

# Linked-calibration script for `from_second`: combines both of `second`'s
# outputs. `run_script(outputs)` receives {output_name: array} for one sample.
_LINK_SCRIPT = (
    "def run_script(outputs):\n"
    "    return outputs['output'] + outputs['pooled']\n"
)

_HAILO_FAST_OPTS = (
    "hailo.compression_level",
    "0",
    "hailo.optimization_level",
    "0",
    "hailo.disable_compilation",
    "True",
)


def _image_inputs(calib_dir: Path) -> list[dict]:
    """The toy net's three image inputs (per-input mean/scale/encoding).

    Upstream stages need image calibration (a directory) -- the multistage
    exporter copies those images to run the stage's inferer for the linked
    calibration of ``third``. The images are constant so quantization stays
    near-lossless.
    """
    calib = {"path": str(calib_dir), "max_images": 8}
    return [
        {
            "name": "bgr",
            "shape": [1, 3, _SIZE, _SIZE],
            "data_type": "float32",
            "mean_values": [10, 20, 30],
            "scale_values": [2, 4, 8],
            "encoding": {"from": "BGR", "to": "BGR"},
            "calibration": calib,
        },
        {
            "name": "rgb",
            "shape": [1, 3, _SIZE, _SIZE],
            "data_type": "float32",
            "mean_values": [5, 6, 7],
            "scale_values": [3, 2, 1],
            "encoding": {"from": "RGB", "to": "BGR"},
            "calibration": calib,
        },
        {
            "name": "gray",
            "shape": [1, 1, _SIZE, _SIZE],
            "data_type": "float32",
            "mean_values": [128],
            "scale_values": [255],
            "encoding": "GRAY",
            "calibration": calib,
        },
    ]


@pytest.fixture(scope="module")
def multistage_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the toy multistage ONNX stages + config; return the config path."""
    workdir = tmp_path_factory.mktemp("toy_multistage")
    first = workdir / "first.onnx"
    second = workdir / "second.onnx"
    third = workdir / "third.onnx"
    build_toy_integration_onnx(first, size=_SIZE)
    build_toy_integration_onnx(second, size=_SIZE)
    build_toy_aggregator_onnx(third, size=_SIZE)

    calib_dir = workdir / "calib"
    calib_dir.mkdir()
    for i in range(8):
        cv2.imwrite(
            str(calib_dir / f"{i}.png"),
            np.full((_SIZE, _SIZE, 3), _CALIB_VALUE, dtype=np.uint8),
        )

    outputs = [{"name": "output"}, {"name": "pooled"}]
    config = {
        "name": "toy_multistage",
        "stages": {
            "first": {
                "input_model": str(first),
                "inputs": _image_inputs(calib_dir),
                "outputs": outputs,
            },
            "second": {
                "input_model": str(second),
                "inputs": _image_inputs(calib_dir),
                "outputs": outputs,
            },
            "third": {
                "input_model": str(third),
                "inputs": [
                    {
                        "name": "from_first",
                        "encoding": "NONE",
                        "calibration": {"stage": "first", "output": "output"},
                    },
                    {
                        "name": "from_second",
                        "encoding": "NONE",
                        "calibration": {
                            "stage": "second",
                            "script": _LINK_SCRIPT,
                        },
                    },
                ],
                "outputs": [{"name": "out"}],
            },
        },
    }
    config_path = workdir / "toy_multistage.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


# Platforms that cannot convert this multistage net, and why.
_CONVERT_XFAIL = {
    "rvc3": "RVC3 quantization does not support multi-input models",
    "hailo": "Hailo quantizer cannot handle the toy net's activation range (shift delta)",
}
_CONVERT_CASES = [
    pytest.param(
        platform,
        marks=[getattr(pytest.mark, platform)]
        + (
            [
                pytest.mark.xfail(
                    reason=_CONVERT_XFAIL[platform],
                    strict=True,
                    raises=SystemExit,
                )
            ]
            if platform in _CONVERT_XFAIL
            else []
        ),
        id=platform,
    )
    for platform in ALL_PLATFORMS
]


@pytest.mark.parametrize("platform", _CONVERT_CASES)
def test_toy_multistage(platform: str, multistage_config: Path):
    output_name = f"_toy-multistage-{platform}"
    extra = _HAILO_FAST_OPTS if platform == "hailo" else ()
    convert(
        Target(platform),
        *extra,
        path=str(multistage_config),
        output_dir=output_name,
        to="native",
        main_stage=MAIN_STAGE,
    )
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    assert any(out_dir.iterdir()), f"no output produced in {out_dir}"


def _golden_pipeline(
    cfg: Config, work: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fp32 reference for the whole pipeline on the constant input.

    Returns ``(final_out, from_first, from_second)`` -- the fp32 upstream
    tensors that feed ``third`` (the same ones the converted ``third`` gets),
    and ``third``'s fp32 output. ``third`` has no preprocessing, so its golden
    is a plain onnxruntime run.
    """
    first, second, third = (
        cfg.stages["first"],
        cfg.stages["second"],
        cfg.stages["third"],
    )
    g_first = golden_reference_outputs(
        Path(first.input_model),
        {i.name: i for i in first.inputs},
        work / "g_first",
        float(_CALIB_VALUE),
    )
    g_second = golden_reference_outputs(
        Path(second.input_model),
        {i.name: i for i in second.inputs},
        work / "g_second",
        float(_CALIB_VALUE),
    )
    from_first = g_first["output"]
    from_second = g_second["output"] + g_second["pooled"]  # the link script

    sess = ort.InferenceSession(
        str(third.input_model), providers=["CPUExecutionProvider"]
    )
    (out,) = sess.run(
        ["out"], {"from_first": from_first, "from_second": from_second}
    )
    return np.asarray(out), from_first, from_second


@pytest.mark.parametrize(
    "platform",
    [
        pytest.param(p, marks=getattr(pytest.mark, p), id=p)
        for p in CORRECTNESS_PLATFORMS
    ],
)
def test_toy_multistage_precision(platform: str, multistage_config: Path):
    target = Target(platform)
    output_name = f"_toy-multistage-prec-{platform}"
    convert(
        target,
        path=str(multistage_config),
        output_dir=output_name,
        to="native",
        main_stage=MAIN_STAGE,
    )

    work = OUTPUTS_DIR / f"{output_name}_work"
    work.mkdir(parents=True, exist_ok=True)
    cfg, _, _ = get_configs(target, str(multistage_config), [])
    golden, from_first_arr, from_second_arr = _golden_pipeline(cfg, work)

    # Feed the converted `third` the same fp32 upstream tensors as the golden.
    # The vendor inferer reads a `.npy` verbatim, so it must already be in the
    # layout that backend expects: SNPE (rvc4) consumes NHWC, OpenVINO (rvc2)
    # NCHW. `from_*_arr` is NCHW [1, C, H, W].
    from_first = work / "from_first.npy"
    from_second = work / "from_second.npy"

    def _as_input(arr: np.ndarray) -> np.ndarray:
        return arr[0].transpose(1, 2, 0) if platform == "rvc4" else arr

    np.save(from_first, _as_input(from_first_arr))
    np.save(from_second, _as_input(from_second_arr))

    third_dir = OUTPUTS_DIR / output_name / "third"
    model_path = locate_converted_model(third_dir, platform)
    inferer = get_inferer(
        target,
        str(model_path),
        work,
        OUTPUTS_DIR / f"{output_name}_infer",
        cfg.stages["third"],
    )
    converted = inferer.infer(
        {"from_first": from_first, "from_second": from_second}
    )

    (conv_out,) = converted.values()
    cos = cosine_similarity(golden, conv_out)
    assert cos >= _THRESHOLD, (
        f"{platform} multistage final output: cosine {cos:.5f} < {_THRESHOLD}"
    )
