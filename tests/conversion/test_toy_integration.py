"""Conversion and precision checks for mixed color and grayscale inputs."""

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
    assert_produced_suffix,
    toy_net_image_inputs,
    write_config,
)
from tests.helpers.onnx_factory import build_toy_integration_onnx
from tests.helpers.onnx_reference import ONNXReferenceInferer
from tests.helpers.platform_options import (
    platform_options,
    superblob_skip_reason,
)
from tests.helpers.platforms import platform_marks
from tests.helpers.precision import (
    RecordProperty,
    check_relative_error,
    locate_converted_model,
)

_SIZE = 64
# Match inference to the calibration distribution to avoid clipping.
_CALIB_MEAN = 100
_CALIB_STD = 20
# Allows about 5x the observed RVC4 INT8 error (0.0018).
_ERROR_LIMIT = 0.01


@dataclass(frozen=True)
class Precision:
    """One precision the toy net is converted at."""

    name: str
    opts: tuple[str, ...] = ()
    xfail: str | None = None
    skip: str | None = None
    error_limit: float | None = None


# Skip slow superblob cases instead of substituting a duplicate blob test.
PRECISIONS: dict[str, list[Precision]] = {
    "rvc2": [
        Precision(
            "blob", ("rvc2.superblob", "False"), error_limit=_ERROR_LIMIT
        ),
        Precision(
            "superblob",
            ("rvc2.superblob", "True"),
            skip=superblob_skip_reason(),
            error_limit=_ERROR_LIMIT,
        ),
    ],
    "rvc3": [
        Precision(
            "non-quant",
            ("rvc3.disable_calibration", "True"),
            error_limit=_ERROR_LIMIT,
        ),
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
            error_limit=_ERROR_LIMIT,
        ),
        Precision(
            "int8-acc",
            ("rvc4.quantization_mode", "INT8_ACCURACY_FOCUSED"),
            error_limit=_ERROR_LIMIT,
        ),
        Precision(
            "int8-int16-mixed",
            ("rvc4.quantization_mode", "INT8_INT16_MIXED"),
            error_limit=_ERROR_LIMIT,
        ),
        Precision(
            "int8-int16-mixed-acc",
            ("rvc4.quantization_mode", "INT8_INT16_MIXED_ACCURACY_FOCUSED"),
            error_limit=_ERROR_LIMIT,
        ),
        # SNPE's CPU runtime cannot execute the FP16 preprocessing Eltwise ops.
        Precision(
            "int16",
            ("rvc4.quantization_mode", "INT16_STANDARD"),
        ),
        Precision("fp16", ("rvc4.quantization_mode", "FP16_STANDARD")),
    ],
    "hailo": [
        # Hailo rejects this activation range: "shift delta > 2".
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
        {"mean": _CALIB_MEAN, "std": _CALIB_STD, "max_images": 8},
        size=_SIZE,
    )
    if with_flag:
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
def inference_images(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    directory = tmp_path_factory.mktemp("toy_noise")
    images = {}
    for seed, name in enumerate(("bgr", "rgb", "gray"), start=1):
        path = directory / f"{name}.png"
        rng = np.random.default_rng(seed)
        noise = rng.normal(_CALIB_MEAN, _CALIB_STD, (_SIZE, _SIZE, 3))
        cv2.imwrite(str(path), np.clip(noise, 0, 255).astype(np.uint8))
        images[name] = path
    return images


def _assert_correct(
    platform: Platform,
    precision: Precision,
    *,
    config_path: Path,
    output_name: str,
    images: dict[str, Path],
    limit: float,
    record_property: RecordProperty,
) -> None:
    cfg, _, _ = get_configs(platform, str(config_path), list(precision.opts))
    stage = next(iter(cfg.stages.values()))
    input_names = [inp.name for inp in stage.inputs]
    assert set(input_names) == set(images)

    reference = ONNXReferenceInferer.from_stage(stage).infer(images)
    model_path = locate_converted_model(
        OUTPUTS_DIR / output_name, platform.value
    )
    inferer = get_inferer(
        platform,
        str(model_path),
        next(iter(images.values())).parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer(images)

    for name, ref in reference.items():
        check_relative_error(
            record_property,
            f"{platform.value} {precision.name} {name}",
            ref,
            converted[name],
            limit,
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
    inference_images: dict[str, Path],
    record_property: RecordProperty,
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
    if platform is Platform.RVC2:
        assert_produced_suffix(output_name, f".{precision.name}")
    else:
        assert_produced(output_name)

    if precision.error_limit is not None:
        _assert_correct(
            platform,
            precision,
            config_path=toy_config,
            output_name=output_name,
            images=inference_images,
            limit=precision.error_limit,
            record_property=record_property,
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
