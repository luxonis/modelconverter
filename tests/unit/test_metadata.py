"""Host-side unit tests for ``modelconverter.utils.metadata``.

Only two of the five ``get_metadata`` backends are exercisable without
vendor tooling:

* ``_get_metadata_onnx`` -- ``onnx`` is installed, so it is covered fully.
* ``_get_metadata_dlc`` via a ``.csv`` path -- the parser reads a
  pre-generated ``snpe-dlc-info`` CSV, so everything except the
  ``snpe-dlc-info`` subprocess branch is reachable with a crafted file.

The IR (``openvino``), TFLite (``tflite``) and Hailo (``hailo_sdk_client``)
backends import vendor libraries that are absent host-side; the tests
below assert only that ``get_metadata`` *dispatches* to them (raising an
``ImportError``), leaving their bodies for Tier-2 coverage.
"""

import sys
from pathlib import Path
from types import ModuleType

import pytest
from onnx import TensorProto

from modelconverter.utils.metadata import (
    Metadata,
    get_metadata,
)
from modelconverter.utils.types import DataType
from tests.helpers.onnx_factory import build_onnx, single_io_onnx

# (suffix, module that the backend fails to import host-side).
VENDOR_SUFFIXES = [
    (".xml", "openvino"),
    (".bin", "openvino"),
    (".tflite", "tflite"),
    (".har", "hailo_sdk_client"),
    (".hef", "hailo_sdk_client"),
]


def _sample_metadata() -> Metadata:
    return Metadata(
        input_shapes={"in": [1, 3, 8, 8]},
        input_dtypes={"in": DataType.FLOAT32},
        output_shapes={"out": [1, 2]},
        output_dtypes={"out": DataType.INT64},
    )


def _assert_standard_io(meta: Metadata, output_dtype: DataType) -> None:
    """The single-in/single-out shapes + dtypes the DLC CSV fixtures share."""
    assert meta.input_shapes == {"input0": [1, 3, 64, 64]}
    assert meta.output_shapes == {"output0": [1, 10]}
    assert meta.input_dtypes["input0"] is DataType.FLOAT32
    assert meta.output_dtypes["output0"] is output_dtype


def test_fields_and_equality():
    # The plain ``Metadata`` dataclass stores the four field mappings, and two
    # independently built instances compare equal.
    meta = _sample_metadata()
    assert meta.input_shapes == {"in": [1, 3, 8, 8]}
    assert meta.output_dtypes["out"] is DataType.INT64
    assert meta == _sample_metadata()


def test_multiple_io_various_dtypes(work_dir: Path):
    """Multiple inputs/outputs with float32/float16/int64/bool."""
    path = build_onnx(
        work_dir / "multi.onnx",
        inputs=[
            ("in_f32", [1, 3, 64, 64], TensorProto.FLOAT),
            ("in_f16", [1, 3, 32, 32], TensorProto.FLOAT16),
            ("in_i64", [1, 5], TensorProto.INT64),
            ("in_bool", [1, 2], TensorProto.BOOL),
        ],
        outputs=[
            ("out_f32", [1, 10], TensorProto.FLOAT),
            ("out_f16", [1, 4], TensorProto.FLOAT16),
            ("out_i64", [2, 3], TensorProto.INT64),
            ("out_bool", [1, 1], TensorProto.BOOL),
        ],
    )
    meta = get_metadata(path)

    assert meta.input_shapes == {
        "in_f32": [1, 3, 64, 64],
        "in_f16": [1, 3, 32, 32],
        "in_i64": [1, 5],
        "in_bool": [1, 2],
    }
    assert meta.input_dtypes == {
        "in_f32": DataType.FLOAT32,
        "in_f16": DataType.FLOAT16,
        "in_i64": DataType.INT64,
        "in_bool": DataType.BOOLEAN,
    }
    assert meta.output_shapes == {
        "out_f32": [1, 10],
        "out_f16": [1, 4],
        "out_i64": [2, 3],
        "out_bool": [1, 1],
    }
    assert meta.output_dtypes == {
        "out_f32": DataType.FLOAT32,
        "out_f16": DataType.FLOAT16,
        "out_i64": DataType.INT64,
        "out_bool": DataType.BOOLEAN,
    }


def test_single_io(work_dir: Path):
    path = single_io_onnx(work_dir / "single.onnx")
    meta = get_metadata(path)
    assert meta.input_shapes == {"input0": [1, 3, 64, 64]}
    assert meta.output_shapes == {"output0": [1, 10]}
    assert meta.input_dtypes["input0"] is DataType.FLOAT32


def test_load_failure_raises(work_dir: Path):
    """A non-ONNX file with an ``.onnx`` suffix raises ``ValueError``."""
    bad = work_dir / "broken.onnx"
    bad.write_text("this is not a protobuf")
    with pytest.raises(ValueError, match="Failed to load ONNX model"):
        get_metadata(bad)


def test_plain_csv(work_dir: Path):
    """A quoted-dimension CSV yields shapes and dtypes for both ends.

    A trailing ``Unconsumed Tensor Name`` block exercises the
    output-section end-marker search.
    """
    csv = (
        "Input Name,Dimensions,Type\n"
        'input0,"1,3,64,64",Float_32\n'
        "Output Name,Dimensions,Type\n"
        'output0,"1,10",Float_32\n'
        "Unconsumed Tensor Name\n"
        "leftover\n"
    )
    path = work_dir / "model.csv"
    path.write_text(csv)
    _assert_standard_io(get_metadata(path), DataType.FLOAT32)


def test_pipe_table_csv(work_dir: Path):
    """The pipe-delimited table branch is normalised into CSV.

    Separator rules (``+---+`` / ``|---|``) are dropped and quoted
    multi-dimension cells survive the ``|`` -> ``,`` rewrite.
    """
    csv = (
        "| Input Name | Dimensions | Type |\n"
        "+------------+------------+------+\n"
        '| input0 | "1,3,64,64" | Float_32 |\n'
        "| Output Name | Dimensions | Type |\n"
        "|-------------|------------|------|\n"
        '| output0 | "1,10" | uInt_8 |\n'
        "Total parameters: 42\n"
    )
    path = work_dir / "piped.csv"
    path.write_text(csv)
    _assert_standard_io(get_metadata(path), DataType.UINT8)


def test_output_section_runs_to_end_without_trailer(work_dir: Path):
    """An output section with no trailing marker extends to EOF.

    Neither ``Unconsumed Tensor Name`` nor ``Total parameters:`` follows
    the outputs, so the section end falls back to the end of the file.
    """
    csv = (
        "Input Name,Dimensions,Type\n"
        'input0,"1,3,64,64",Float_32\n'
        "Output Name,Dimensions,Type\n"
        'output0,"1,10",Float_32\n'
    )
    path = work_dir / "no_trailer.csv"
    path.write_text(csv)
    meta = get_metadata(path)
    assert meta.input_shapes == {"input0": [1, 3, 64, 64]}
    assert meta.output_shapes == {"output0": [1, 10]}


def test_missing_input_section_raises(work_dir: Path):
    """A CSV with no ``Input Name`` section is skipped, leaving the
    inputs unset -- ``Metadata`` then rejects the partial result."""
    csv = 'Output Name,Dimensions,Type\noutput0,"1,10",Float_32\n'
    path = work_dir / "no_inputs.csv"
    path.write_text(csv)
    with pytest.raises(TypeError):
        get_metadata(path)


def test_unsupported_suffix(work_dir: Path):
    with pytest.raises(ValueError, match="Unsupported model format"):
        get_metadata(work_dir / "model.pt")


@pytest.mark.parametrize(("suffix", "vendor_mod"), VENDOR_SUFFIXES)
def test_vendor_dispatch(work_dir: Path, suffix: str, vendor_mod: str):
    """Vendor routes dispatch correctly, raising ``ImportError``.

    When the vendor library is genuinely absent (the host case) the
    backend fails at its top-level import, which proves dispatch was
    reached. If the library is present the body itself would run, so
    the test is skipped as out of host scope.
    """
    try:
        __import__(vendor_mod)
    except ImportError:
        pass
    else:
        pytest.skip(f"`{vendor_mod}` importable; backend body out of scope")

    path = work_dir / f"model{suffix}"
    path.write_bytes(b"dummy")
    with pytest.raises(ImportError):
        get_metadata(path)


def test_ir_load_failure_raises(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    class _Core:
        def read_model(self, model: str, weights: str) -> None:
            raise RuntimeError("bad IR")

    openvino = ModuleType("openvino")
    runtime = ModuleType("openvino.runtime")
    runtime.Core = _Core  # type: ignore[attr-defined]
    openvino.runtime = runtime  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openvino", openvino)
    monkeypatch.setitem(sys.modules, "openvino.runtime", runtime)

    with pytest.raises(ValueError, match="Failed to load IR model"):
        get_metadata(work_dir / "model.xml")


def test_tflite_missing_subgraph_raises(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    class _Root:
        def Subgraphs(self, i: int) -> None:
            return None

    class _Model:
        @staticmethod
        def GetRootAsModel(data: bytes, offset: int) -> "_Root":
            return _Root()

    tflite = ModuleType("tflite")
    tflite.Model = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tflite", tflite)

    path = work_dir / "model.tflite"
    path.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="Failed to load TFLite model"):
        get_metadata(path)
