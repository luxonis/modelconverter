"""Host-side unit tests for how an ``Inferer`` claims its destination.

Inference results are written to ``output/<output_dir>`` on the host,
the same tree the conversions write into. A rerun therefore has to clear
its own results without ever touching a directory it did not produce,
which it decides by the marker file it leaves behind.

``Inferer`` is abstract and every concrete subclass pulls in a
vendor-heavy module, so the destination handling in ``__post_init__`` is
exercised through a minimal stand-in subclass.
"""

from pathlib import Path

import numpy as np
import pytest

from modelconverter.platforms.base_inferer import Inferer
from modelconverter.utils import ModelconverterException
from modelconverter.utils.constants import INFERENCE_MARKER


class _StubInferer(Inferer):
    """The smallest concrete ``Inferer``; the destination handling runs
    before anything vendor-specific would."""

    def setup(self) -> None: ...

    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]:
        return {}


def _infer_into(dest: Path) -> _StubInferer:
    return _StubInferer(
        model_path=Path("model.dlc"),
        src=Path("src"),
        dest=dest,
        in_shapes={},
        in_dtypes={},
        out_shapes={},
        out_dtypes={},
        resize_method={},
        encoding={},
    )


def test_missing_destination_is_created_and_marked(work_dir: Path):
    dest = work_dir / "output" / "results"

    _infer_into(dest)

    assert dest.is_dir()
    assert (dest / INFERENCE_MARKER).is_file()


def test_empty_destination_is_accepted(work_dir: Path):
    """An empty directory holds nothing that could be lost, so the
    missing marker must not block it -- the user may well have created
    the directory themselves."""
    dest = work_dir / "output" / "results"
    dest.mkdir(parents=True)

    _infer_into(dest)

    assert (dest / INFERENCE_MARKER).is_file()


def test_own_results_are_replaced(work_dir: Path):
    """A rerun starts from a clean directory, so stale outputs from the
    previous run cannot be mistaken for this run's."""
    dest = work_dir / "output" / "results"
    dest.mkdir(parents=True)
    (dest / INFERENCE_MARKER).touch()
    stale = dest / "output0"
    stale.mkdir()
    (stale / "0.npy").write_bytes(b"stale")

    _infer_into(dest)

    assert not stale.exists()
    assert (dest / INFERENCE_MARKER).is_file()


def test_foreign_directory_is_refused(work_dir: Path):
    """``output/`` also holds the conversion results, so an
    ``--output-dir`` that collides with one must fail instead of
    deleting it."""
    dest = work_dir / "output" / "mnist_to_rvc4_2026_01_01_00_00_00"
    dest.mkdir(parents=True)
    converted = dest / "mnist.dlc"
    converted.write_bytes(b"converted model")

    with pytest.raises(ModelconverterException, match="is not empty"):
        _infer_into(dest)

    assert converted.read_bytes() == b"converted model"


def test_file_destination_is_refused(work_dir: Path):
    dest = work_dir / "output" / "results"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"not a directory")

    with pytest.raises(ModelconverterException, match="not a directory"):
        _infer_into(dest)

    assert dest.read_bytes() == b"not a directory"
