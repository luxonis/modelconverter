"""Toy conv-net numeric-fidelity tests -- **all platforms**.

``build_toy_integration_onnx`` covers the preprocessing surface but is a poor
fidelity subject: its output is essentially the preprocessed input, and its wide
per-input scale spread makes the Hailo quantizer reject it, so the toy
integration precision check only covers rvc2/rvc4.

This uses the well-conditioned single-input conv toy instead, which every backend
including Hailo can quantize and reproduce near-losslessly. It still carries
mean/scale + channel reversal in the config, so the full baked-preprocessing path
is exercised -- the difference is only that the graph does real quantizable work.
RVC3, RVC4, and Hailo additionally convert the toy with preprocessing stored in
the NN Archive, feed the bare model an equivalent preprocessed tensor, and compare
both quantized paths with each other and with the original ONNX.

Run inside the platform Docker image, e.g.::

    modelconverter shell hailo --dev --no-gpu -c 'python -m pytest -k toy_precision'
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import extract_preprocessing, get_configs
from modelconverter.platforms.base_exporter import Exporter
from modelconverter.platforms.getters import get_inferer
from modelconverter.utils.config import ImageCalibrationConfig
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Platform
from tests.helpers.conversion import write_toy_conv_config
from tests.helpers.onnx_reference import ONNXReferenceInferer
from tests.helpers.platform_options import platform_options
from tests.helpers.platforms import platform_params
from tests.helpers.precision import (
    cosine_similarity,
    golden_reference_outputs,
    locate_converted_model,
)

_SIZE = 32
# Constant the fidelity comparison runs on, inside the calibration range so the
# quantized model represents it well.
_INFER_VALUE = 100
_THRESHOLD = 0.9
# Both quantized paths run the same graph on the same numbers, so they have to
# agree with each other much more closely than either does with the ONNX.
_EQUIVALENCE_THRESHOLD = 0.99

# Hailo: skip the slow HEF compile -- the check runs `SDK_QUANTIZED` inference on
# the quantized HAR, so compilation is unneeded.
_PLATFORM_OPTS: dict[str, tuple[str, ...]] = {
    "hailo": ("hailo.disable_compilation", "True"),
}


@pytest.fixture(scope="module")
def toy_conv_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_toy_conv_config(
        tmp_path_factory.mktemp("toy_precision"), size=_SIZE
    )


@pytest.fixture(scope="module")
def constant_image(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("toy_prec_const") / "const.png"
    cv2.imwrite(
        str(path), np.full((_SIZE, _SIZE, 3), _INFER_VALUE, dtype=np.uint8)
    )
    return path


def _to_nchw(arr: np.ndarray, platform_name: str) -> np.ndarray:
    """Normalize a platform's spatial output to NCHW.

    The Hailo inferer returns spatial outputs channels-last; the others,
    and the golden reference, are already NCHW.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and platform_name == "hailo":
        return arr.transpose(2, 0, 1)[np.newaxis]
    return arr


@pytest.mark.parametrize("platform_name", platform_params())
def test_toy_precision(
    platform_name: str, toy_conv_config: Path, constant_image: Path
):
    platform = Platform(platform_name)
    opts = (
        *platform_options(platform),
        *_PLATFORM_OPTS.get(platform_name, ()),
    )
    output_name = f"_toy-prec-{platform_name}"

    convert(
        platform,
        *opts,
        path=str(toy_conv_config),
        output_dir=output_name,
        to="native",
    )

    cfg, _, _ = get_configs(platform, str(toy_conv_config), list(opts))
    stage = next(iter(cfg.stages.values()))

    reference = golden_reference_outputs(
        Path(stage.input_model),
        {inp.name: inp for inp in stage.inputs},
        OUTPUTS_DIR / f"{output_name}_golden",
        float(_INFER_VALUE),
    )
    model_path = locate_converted_model(
        OUTPUTS_DIR / output_name, platform_name
    )
    inferer = get_inferer(
        platform,
        str(model_path),
        constant_image.parent,
        OUTPUTS_DIR / f"{output_name}_infer",
        stage,
    )
    converted = inferer.infer({"img": constant_image})

    for (name, ref), conv in zip(
        reference.items(), converted.values(), strict=True
    ):
        cos = cosine_similarity(ref, _to_nchw(conv, platform_name))
        assert cos >= _THRESHOLD, (
            f"{platform_name} output {name!r}: cosine {cos:.5f} < {_THRESHOLD}"
        )


@pytest.mark.parametrize(
    "platform_name", platform_params(("rvc3", "rvc4", "hailo"))
)
def test_externalized_calibration_matches_embedded_model(
    platform_name: str,
    toy_conv_config: Path,
    tmp_path: Path,
) -> None:
    """Quantized embedded and archive-preprocessed models remain equivalent."""
    platform = Platform(platform_name)
    options = (
        *platform_options(platform),
        *_PLATFORM_OPTS.get(platform_name, ()),
    )
    embedded_name = f"_toy-prec-{platform_name}-embedded"
    externalized_name = f"_toy-prec-{platform_name}-externalized"

    convert(
        platform,
        *options,
        path=str(toy_conv_config),
        output_dir=embedded_name,
        to="native",
    )
    convert(
        platform,
        *options,
        path=str(toy_conv_config),
        output_dir=externalized_name,
        to="nn_archive",
        archive_preprocess=True,
    )

    cfg, _, _ = get_configs(platform, str(toy_conv_config), list(options))
    stage = next(iter(cfg.stages.values()))
    runtime_image = tmp_path / "runtime.png"
    cv2.imwrite(
        str(runtime_image),
        np.tile(np.array([80, 100, 120], dtype=np.uint8), (_SIZE, _SIZE, 1)),
    )
    reference = ONNXReferenceInferer.from_stage(stage).infer(runtime_image)

    externalized_cfg = cfg.model_copy(deep=True)
    extract_preprocessing(externalized_cfg)
    externalized_stage = next(iter(externalized_cfg.stages.values()))
    externalized_input = externalized_stage.inputs[0]
    calibration = externalized_input.calibration
    assert isinstance(calibration, ImageCalibrationConfig)
    model_array, layout = Exporter._read_calibration_file(
        externalized_input, calibration, runtime_image
    )
    assert layout == "HWC"
    if platform in {Platform.RVC3, Platform.HAILO}:
        model_array = model_array.transpose(2, 0, 1)
    model_input = tmp_path / "model-domain.npy"
    np.save(model_input, model_array)

    embedded_model = locate_converted_model(
        OUTPUTS_DIR / embedded_name, platform.value
    )
    externalized_model = locate_converted_model(
        OUTPUTS_DIR / externalized_name, platform.value
    )
    embedded = get_inferer(
        platform,
        str(embedded_model),
        runtime_image.parent,
        OUTPUTS_DIR / f"{embedded_name}_infer",
        stage,
    ).infer({"img": runtime_image})
    externalized = get_inferer(
        platform,
        str(externalized_model),
        model_input.parent,
        OUTPUTS_DIR / f"{externalized_name}_infer",
        externalized_stage,
    ).infer({"img": model_input})

    assert len(embedded) == len(externalized) == len(reference)
    for (name, expected), embedded_value, externalized_value in zip(
        reference.items(),
        embedded.values(),
        externalized.values(),
        strict=True,
    ):
        embedded_output = _to_nchw(embedded_value, platform.value)
        externalized_output = _to_nchw(externalized_value, platform.value)
        checks = (
            (
                "embedded",
                cosine_similarity(expected, embedded_output),
                _THRESHOLD,
            ),
            (
                "externalized",
                cosine_similarity(expected, externalized_output),
                _THRESHOLD,
            ),
            (
                "embedded vs externalized",
                cosine_similarity(embedded_output, externalized_output),
                _EQUIVALENCE_THRESHOLD,
            ),
        )
        for compared, cos, threshold in checks:
            assert cos >= threshold, (
                f"{platform_name} {compared} output {name!r}: "
                f"cosine {cos:.5f} < {threshold}"
            )
