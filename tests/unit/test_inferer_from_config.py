"""Host-side unit tests for ``Inferer.from_config``.

``Inferer`` is abstract and every concrete subclass lives in a
vendor-heavy module (RVC2 needs ``openvino``, Hailo needs
``hailo_sdk_client``), so the mapping from a ``SingleStageConfig`` onto
the per-input dictionaries is exercised through a minimal stand-in
subclass instead.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from luxonis_ml.typing import ParamValue

from modelconverter.platforms.base_inferer import Inferer
from modelconverter.utils.config import Config
from modelconverter.utils.types import Encoding, ResizeMethod
from tests.helpers.onnx_factory import single_io_onnx


class _StubInferer(Inferer):
    """The smallest concrete ``Inferer``; ``from_config`` needs no
    more.
    """

    def setup(self) -> None: ...

    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]:
        return {}


def _build(
    work_dir: Path, shape: list[int], **overrides: ParamValue
) -> _StubInferer:
    """Build an inferer from a single-input model of the given shape."""
    model = single_io_onnx(
        work_dir / "m.onnx",
        shape=shape,
    ).resolve()
    config = Config.get_config(
        None, {"input_model": str(model), "shape": shape, **overrides}
    )
    stage = next(iter(config.stages.values()))
    return _StubInferer.from_config(
        str(model), work_dir / "src", work_dir / "dest", stage
    )


@pytest.mark.parametrize(
    ("shape", "overrides", "expected"),
    [
        ([1, 3, 64, 64], {}, Encoding.BGR),
        ([1, 3, 64, 64], {"encoding": "RGB"}, Encoding.RGB),
        ([1, 3, 64, 64], {"encoding": "GRAY"}, Encoding.GRAY),
        ([1, 1, 64, 64], {}, Encoding.GRAY),
    ],
)
def test_encoding_follows_the_config(
    work_dir: Path,
    shape: list[int],
    overrides: Mapping[str, ParamValue],
    expected: Encoding,
):
    """``encoding.to`` is passed through whatever the calibration is.

    None of these inputs configure calibration, so they all keep the
    default ``RandomCalibrationConfig``. A grayscale input must still
    map to ``GRAY`` rather than being forced to ``BGR``.
    """
    inferer = _build(work_dir, shape, **overrides)
    assert inferer.encoding == {"input0": expected}


def test_grayscale_input_is_not_forced_to_bgr(work_dir: Path):
    """A 1-channel input is detected as grayscale by the config, and
    that has to survive into the inferer.

    ``read_image`` would otherwise decode a grayscale image as
    3-channel BGR.
    """
    inferer = _build(work_dir, [1, 1, 64, 64])
    assert inferer.encoding["input0"] is Encoding.GRAY


def test_layout_is_taken_from_the_config(work_dir: Path):
    inferer = _build(work_dir, [1, 3, 64, 64])
    assert inferer.layout == {"input0": "NCHW"}


def test_resize_method_defaults_without_image_calibration(work_dir: Path):
    """Only ``ImageCalibrationConfig`` carries a resize method."""
    inferer = _build(work_dir, [1, 3, 64, 64])
    assert inferer.resize_method == {"input0": ResizeMethod.RESIZE}


def test_shapes_and_dtypes_are_mapped(work_dir: Path):
    inferer = _build(work_dir, [1, 3, 64, 64])
    assert inferer.in_shapes == {"input0": [1, 3, 64, 64]}
    assert inferer.out_shapes == {"output0": [1, 10]}


def test_the_stand_in_reports_no_outputs(work_dir: Path):
    """The stand-in has no vendor runtime, so its forward pass produces
    nothing; `from_config` needs no more than that.
    """
    assert _build(work_dir, [1, 3, 64, 64]).infer({}) == {}
