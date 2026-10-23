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

import pytest

from modelconverter.platforms.base_exporter import Exporter
from modelconverter.utils.config import InputConfig, RandomCalibrationConfig
from modelconverter.utils.types import DataType


def _prepare(
    tmp_path: Path,
    shape: list[int],
    layout: str | None,
    *,
    encoding: str | None = None,
    data_type: DataType = DataType.FLOAT32,
) -> list[str]:
    """Run the calibration prep for one input, returning the written suffixes."""
    data: dict = {"name": "x", "shape": shape}
    if layout is not None:
        data["layout"] = layout
    if encoding is not None:
        data["encoding"] = encoding
    inp = InputConfig.model_validate(data)
    inp.calibration = RandomCalibrationConfig(
        max_images=2, data_type=data_type
    )

    # The method only touches `_inputs` and `intermediate_outputs_dir`, so a
    # lightweight stand-in stands in for a fully constructed exporter.
    stub = SimpleNamespace(
        _inputs={"x": inp}, intermediate_outputs_dir=tmp_path
    )
    Exporter._prepare_random_calibration_data(stub)  # pyright: ignore[reportArgumentType]
    return sorted(p.suffix for p in (tmp_path / "random" / "x").iterdir())


@pytest.mark.parametrize(
    ("shape", "layout", "encoding", "suffix"),
    [
        # 4D NCHW image: channel axis is known from the layout (the path the
        # conversion tests already cover) -> written as a `.png`.
        ([1, 3, 8, 8], None, None, ".png"),
        # Explicit color encoding keeps the legacy channels-first image path
        # for a rank-3 tensor with a C-less layout.
        ([3, 4, 5], "TNF", "RGB", ".png"),
        # Without an explicit color contract, the same nonstandard layout is
        # treated as a raw tensor.
        ([3, 4, 5], "TNF", None, ".npy"),
        # Rank-1 feature vector: not image-like -> raw `.npy`.
        ([8], None, None, ".npy"),
        # Batched (N > 1) tensor: not image-like -> raw `.npy`.
        ([2, 3, 8, 8], None, None, ".npy"),
        # Raw tensors use NumPy even when their rank resembles an image.
        ([1, 32], "NC", "NONE", ".npy"),
    ],
)
def test_random_calibration_output_format(
    tmp_path: Path,
    shape: list[int],
    layout: str | None,
    encoding: str | None,
    suffix: str,
):
    assert _prepare(tmp_path, shape, layout, encoding=encoding) == [
        suffix,
        suffix,
    ]


@pytest.mark.parametrize("data_type", [DataType.BOOLEAN, DataType.STRING])
def test_image_samples_without_a_numeric_dtype_are_saved_by_numpy(
    tmp_path: Path, data_type: DataType
):
    """OpenCV writes only integer and float pixels."""
    assert _prepare(tmp_path, [1, 3, 8, 8], None, data_type=data_type) == [
        ".npy",
        ".npy",
    ]
