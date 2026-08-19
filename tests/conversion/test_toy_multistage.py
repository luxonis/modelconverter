r"""Toy multistage conversion + correctness tests.

A small, fully-controllable multistage net built from the toy pieces::

    first  (toy net)  ->\\
                         > third (from_first * from_second + bias)
    second (toy net)  ->/

``third``'s two inputs come from *linked calibration*, covering both linking
options: ``from_first`` from a direct output link, ``from_second`` from a script
that combines ``second``'s two outputs. Stage order in the config is topological,
so the upstream stages are converted before ``third``'s linked calibration runs
their inferers.

Conversion is checked on every platform; correctness against an fp32 golden
pipeline on rvc2/rvc4, matching the single-stage toy -- their preprocessing goes
through the golden path and their DLC/IR runs on the CI CPU backend.

Run inside the platform Docker image, e.g.::

    modelconverter shell rvc4 --dev -c 'python -m pytest -k toy_multistage'
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.packages.getters import get_inferer
from modelconverter.utils.config import Config
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target
from tests.helpers.conversion import (
    HAILO_FAST_OPTS,
    assert_produced,
    toy_net_image_inputs,
    write_config,
)
from tests.helpers.onnx_factory import (
    build_toy_aggregator_onnx,
    build_toy_integration_onnx,
)
from tests.helpers.platforms import platform_params
from tests.helpers.precision import (
    cosine_similarity,
    golden_reference_outputs,
    locate_converted_model,
)
from tests.helpers.target_options import target_options

_SIZE = 64
_CALIB_VALUE = 100
_THRESHOLD = 0.9

CORRECTNESS_PLATFORMS = ("rvc2", "rvc4")
MAIN_STAGE = "third"

# Linked-calibration script for `from_second`. `run_script(outputs)` receives
# {output_name: array} for one sample.
_LINK_SCRIPT = (
    "def run_script(outputs):\n"
    "    return outputs['output'] + outputs['pooled']\n"
)

_CONVERT_XFAILS = {
    "rvc3": "RVC3 quantization does not support multi-input models",
    "hailo": "Hailo quantizer cannot handle the toy net's activation range",
}


@pytest.fixture(scope="module")
def multistage_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workdir = tmp_path_factory.mktemp("toy_multistage")
    first = build_toy_integration_onnx(workdir / "first.onnx", size=_SIZE)
    second = build_toy_integration_onnx(workdir / "second.onnx", size=_SIZE)
    third = build_toy_aggregator_onnx(workdir / "third.onnx", size=_SIZE)

    # Upstream stages need image calibration (a directory): the multistage
    # exporter copies those images to run each stage's inferer for `third`'s
    # linked calibration. Constant images keep quantization near-lossless.
    calib_dir = workdir / "calib"
    calib_dir.mkdir()
    for i in range(8):
        cv2.imwrite(
            str(calib_dir / f"{i}.png"),
            np.full((_SIZE, _SIZE, 3), _CALIB_VALUE, dtype=np.uint8),
        )

    def toy_stage(model: Path) -> dict:
        return {
            "input_model": str(model),
            "inputs": toy_net_image_inputs(
                {"path": str(calib_dir), "max_images": 8}, size=_SIZE
            ),
            "outputs": [{"name": "output"}, {"name": "pooled"}],
        }

    return write_config(
        workdir,
        "toy_multistage",
        {
            "name": "toy_multistage",
            "stages": {
                "first": toy_stage(first),
                "second": toy_stage(second),
                "third": {
                    "input_model": str(third),
                    "inputs": [
                        {
                            "name": "from_first",
                            "encoding": "NONE",
                            "calibration": {
                                "stage": "first",
                                "output": "output",
                            },
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
        },
    )


@pytest.mark.parametrize("platform", platform_params(xfails=_CONVERT_XFAILS))
def test_toy_multistage(platform: str, multistage_config: Path):
    target = Target(platform)
    output_name = f"_toy-multistage-{platform}"
    extra = HAILO_FAST_OPTS if platform == "hailo" else ()
    convert(
        target,
        *target_options(target),
        *extra,
        path=str(multistage_config),
        output_dir=output_name,
        to="native",
        main_stage=MAIN_STAGE,
    )
    assert_produced(output_name)


def _golden_pipeline(
    cfg: Config, work: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fp32 reference for the whole pipeline on the constant input.

    Returns ``third``'s fp32 output plus the two fp32 upstream tensors that feed
    it -- the same ones the converted ``third`` gets. ``third`` has no
    preprocessing, so its golden is a plain onnxruntime run.
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

    session = ort.InferenceSession(
        str(third.input_model), providers=["CPUExecutionProvider"]
    )
    (out,) = session.run(
        ["out"], {"from_first": from_first, "from_second": from_second}
    )
    return np.asarray(out), from_first, from_second


@pytest.mark.parametrize("platform", platform_params(CORRECTNESS_PLATFORMS))
def test_toy_multistage_precision(platform: str, multistage_config: Path):
    target = Target(platform)
    options = target_options(target)
    output_name = f"_toy-multistage-prec-{platform}"
    convert(
        target,
        *options,
        path=str(multistage_config),
        output_dir=output_name,
        to="native",
        main_stage=MAIN_STAGE,
    )

    work = OUTPUTS_DIR / f"{output_name}_work"
    work.mkdir(parents=True, exist_ok=True)
    cfg, _, _ = get_configs(target, str(multistage_config), list(options))
    golden, from_first_arr, from_second_arr = _golden_pipeline(cfg, work)

    # Feed the converted `third` the same fp32 upstream tensors as the golden.
    # The vendor inferer reads a `.npy` verbatim, so it must already be in the
    # layout that backend expects: SNPE (rvc4) consumes NHWC, OpenVINO (rvc2)
    # NCHW, and `from_*_arr` is NCHW.
    def as_input(arr: np.ndarray) -> np.ndarray:
        return arr[0].transpose(1, 2, 0) if platform == "rvc4" else arr

    from_first = work / "from_first.npy"
    from_second = work / "from_second.npy"
    np.save(from_first, as_input(from_first_arr))
    np.save(from_second, as_input(from_second_arr))

    model_path = locate_converted_model(
        OUTPUTS_DIR / output_name / "third", platform
    )
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
