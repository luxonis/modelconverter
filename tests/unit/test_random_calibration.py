"""Random-calibration data prep for unusual input shapes / layouts.

``Exporter._prepare_random_calibration_data`` writes one sample per input,
picking a ``.png`` for image-like tensors and a ``.npy`` for the rest, and
transposing channels-first data for the image writer. The 4D ``NCHW`` inputs the
conversion tests use only reach the "channel axis known from the layout" path;
the cases below cover the fallbacks it misses.

Driving the method directly keeps this deterministic and host-side -- the random
pixel values never affect which branch runs.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from modelconverter.platforms.base_exporter import Exporter
from modelconverter.utils.config import InputConfig, RandomCalibrationConfig


def _prepare(
    tmp_path: Path, shape: list[int], layout: str | None, *, raw: bool = False
) -> list[str]:
    """Run the calibration prep for one input, returning the written suffixes."""
    data: dict = {"name": "x", "shape": shape}
    if layout is not None:
        data["layout"] = layout
    if raw:
        data["encoding"] = {"from": "NONE", "to": "NONE"}
    inp = InputConfig.model_validate(data)
    inp.calibration = RandomCalibrationConfig(max_images=2)

    # The method only touches `inputs` and `intermediate_outputs_dir`, so a
    # lightweight stand-in stands in for a fully constructed exporter.
    stub = SimpleNamespace(
        inputs={"x": inp}, intermediate_outputs_dir=tmp_path
    )
    Exporter._prepare_random_calibration_data(stub)  # type: ignore[arg-type]
    return sorted(p.suffix for p in (tmp_path / "random" / "x").iterdir())


@pytest.mark.parametrize(
    ("shape", "layout", "raw", "suffix"),
    [
        # 4D NCHW image: channel axis is known from the layout (the path the
        # conversion tests already cover) -> written as a `.png`.
        ([1, 3, 8, 8], None, False, ".png"),
        # Rank-3 tensor with a C-less layout: the channels-first heuristic
        # (`shape[0] in {1, 3}`) transposes to HWC before the `.png` write.
        ([3, 4, 5], "TNF", False, ".png"),
        # Rank-1 feature vector: not image-like -> raw `.npy`.
        ([8], None, False, ".npy"),
        # Batched (N > 1) tensor: not image-like -> raw `.npy`.
        ([2, 3, 8, 8], None, False, ".npy"),
        # Raw tensors use NumPy even when their rank resembles an image.
        ([1, 32], "NC", True, ".npy"),
    ],
)
def test_random_calibration_output_format(
    tmp_path: Path,
    shape: list[int],
    layout: str | None,
    raw: bool,
    suffix: str,
):
    np.random.seed(0)
    assert _prepare(tmp_path, shape, layout, raw=raw) == [suffix, suffix]
