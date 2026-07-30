"""Host-side unit tests for ``modelconverter.utils.types``.

``DataType`` is mostly a bundle of lookup tables between vendor dtype
spellings. The tables below describe every ``from_*`` converter in one
place -- its accepted spellings, a value it must reject and the error it
must raise -- so a new backend is a single table entry rather than a new
pair of near-identical test functions.

``from_dai_dtype`` and ``from_tensorflow_dtype`` stay separate: the
former needs ``depthai`` imported lazily (so a missing wheel fails one
test rather than collection), the latter a faked ``tflite`` module.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from onnx.onnx_pb import TensorProto

from modelconverter.utils.types import DataType, InputFileType


class _Converter(NamedTuple):
    """One ``DataType.from_*`` classmethod and everything it accepts."""

    convert: Callable[[Any], DataType]
    error: str
    unsupported: Any
    cases: dict[Any, DataType]


CONVERTERS = {
    "onnx": _Converter(
        DataType.from_onnx_dtype,
        "Unsupported ONNX",
        TensorProto.UNDEFINED,
        {
            TensorProto.BFLOAT16: DataType.BFLOAT16,
            TensorProto.FLOAT16: DataType.FLOAT16,
            TensorProto.FLOAT: DataType.FLOAT32,
            TensorProto.DOUBLE: DataType.FLOAT64,
            TensorProto.INT4: DataType.INT4,
            TensorProto.INT8: DataType.INT8,
            TensorProto.INT16: DataType.INT16,
            TensorProto.INT32: DataType.INT32,
            TensorProto.INT64: DataType.INT64,
            TensorProto.UINT4: DataType.UINT4,
            TensorProto.UINT8: DataType.UINT8,
            TensorProto.UINT16: DataType.UINT16,
            TensorProto.UINT32: DataType.UINT32,
            TensorProto.UINT64: DataType.UINT64,
            TensorProto.BOOL: DataType.BOOLEAN,
            TensorProto.STRING: DataType.STRING,
        },
    ),
    "numpy": _Converter(
        DataType.from_numpy_dtype,
        "Unsupported numpy",
        np.complex128,
        {
            np.float16: DataType.FLOAT16,
            np.float32: DataType.FLOAT32,
            np.float64: DataType.FLOAT64,
            np.int8: DataType.INT8,
            np.int16: DataType.INT16,
            np.int32: DataType.INT32,
            np.int64: DataType.INT64,
            np.uint8: DataType.UINT8,
            np.uint16: DataType.UINT16,
            np.uint32: DataType.UINT32,
            np.uint64: DataType.UINT64,
        },
    ),
    "ir_ie": _Converter(
        DataType.from_ir_ie_dtype,
        "Unsupported IR",
        "nope",
        {
            "FP16": DataType.FLOAT16,
            "FP32": DataType.FLOAT32,
            "FP64": DataType.FLOAT64,
            "I8": DataType.INT8,
            "I16": DataType.INT16,
            "I32": DataType.INT32,
            "I64": DataType.INT64,
            "U8": DataType.UINT8,
            "U16": DataType.UINT16,
            "U32": DataType.UINT32,
            "U64": DataType.UINT64,
            "BOOL": DataType.BOOLEAN,
        },
    ),
    "ir_runtime": _Converter(
        DataType.from_ir_runtime_dtype,
        "Unsupported IR runtime",
        "nope",
        {
            "f16": DataType.FLOAT16,
            "f32": DataType.FLOAT32,
            "f64": DataType.FLOAT64,
            "i8": DataType.INT8,
            "i16": DataType.INT16,
            "i32": DataType.INT32,
            "i64": DataType.INT64,
            "u8": DataType.UINT8,
            "u16": DataType.UINT16,
            "u32": DataType.UINT32,
            "u64": DataType.UINT64,
            "boolean": DataType.BOOLEAN,
        },
    ),
    "dlc": _Converter(
        DataType.from_dlc_dtype,
        "Unsupported DLC",
        "nope",
        {
            "Float_16": DataType.FLOAT16,
            "Float_32": DataType.FLOAT32,
            "Float_64": DataType.FLOAT64,
            "Int_8": DataType.INT8,
            "Int_16": DataType.INT16,
            "Int_32": DataType.INT32,
            "Int_64": DataType.INT64,
            "uInt_8": DataType.UINT8,
            "uInt_16": DataType.UINT16,
            "uInt_32": DataType.UINT32,
            "uInt_64": DataType.UINT64,
            "uFxp_8": DataType.UFXP8,
            "uFxp_16": DataType.UFXP16,
            "uFxp_32": DataType.UFXP32,
            "Fxp_8": DataType.FXP8,
            "Fxp_16": DataType.FXP16,
            "Fxp_32": DataType.FXP32,
        },
    ),
    "hubai": _Converter(
        DataType.from_hubai_dtype,
        "Unsupported HubAI",
        "nope",
        {
            "INT8": DataType.INT8,
            "INT32": DataType.INT32,
            "FP16": DataType.FLOAT16,
            "FP32": DataType.FLOAT32,
        },
    ),
}

STRING_CONVERTERS = ["ir_ie", "ir_runtime", "dlc", "hubai"]


@pytest.mark.parametrize(
    ("converter", "raw", "expected"),
    [
        pytest.param(
            entry.convert, raw, expected, id=f"{name}-{expected.value}"
        )
        for name, entry in CONVERTERS.items()
        for raw, expected in entry.cases.items()
    ],
)
def test_from_dtype_supported(
    converter: Callable[[Any], DataType], raw: Any, expected: DataType
):
    assert converter(raw) is expected


@pytest.mark.parametrize("name", CONVERTERS, ids=list(CONVERTERS))
def test_from_dtype_unsupported(name: str):
    entry = CONVERTERS[name]
    with pytest.raises(ValueError, match=entry.error):
        entry.convert(entry.unsupported)


@pytest.mark.parametrize("name", STRING_CONVERTERS)
@given(spelling=st.text(max_size=12))
def test_unknown_spellings_are_rejected(name: str, spelling: str):
    """Any spelling outside the table raises, rather than mis-mapping.

    The vendor tables are keyed by exact strings, so a near-miss such as
    ``"fp32"`` for ``"FP32"`` must be an error and never silently fall
    through to a neighbouring dtype.
    """
    entry = CONVERTERS[name]
    assume(spelling not in entry.cases)
    with pytest.raises(ValueError, match=entry.error):
        entry.convert(spelling)


# ``depthai`` is installed host-side, so cover it directly.
FROM_DAI_DTYPE_CASES = [
    ("FP16", DataType.FLOAT16),
    ("FP32", DataType.FLOAT32),
    ("FP64", DataType.FLOAT64),
    ("I8", DataType.INT8),
    ("INT", DataType.INT32),
    ("U8F", DataType.UFXP8),
    ("U16F", DataType.UFXP16),
]


@pytest.mark.parametrize(("member", "expected"), FROM_DAI_DTYPE_CASES)
def test_from_dai_dtype_supported(member: str, expected: DataType):
    import depthai as dai

    dtype = getattr(dai.TensorInfo.DataType, member)
    assert DataType.from_dai_dtype(dtype) is expected


def test_from_dai_dtype_unsupported():
    # No unmapped ``dai`` enum member exists, so pass a value that is
    # provably absent from the map to hit the error branch.
    with pytest.raises(ValueError, match="Unsupported DepthAI"):
        DataType.from_dai_dtype(None)  # type: ignore[arg-type]


class _FakeTensorType:
    """Stand-in for ``tflite.TensorType.TensorType``.

    ``tflite`` is not installed host-side; the real enum only supplies
    integer keys, so distinct ints reproduce it faithfully.
    """

    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    UINT8 = 3
    INT64 = 4
    STRING = 5
    BOOL = 6
    INT16 = 7
    FLOAT64 = 11
    UINT32 = 12
    UINT16 = 13


@pytest.fixture
def fake_tflite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``tflite`` package so the import inside
    ``from_tensorflow_dtype`` resolves without the real dependency."""
    import sys
    import types as _types

    tflite_mod = _types.ModuleType("tflite")
    tensortype_mod = _types.ModuleType("tflite.TensorType")
    tensortype_mod.TensorType = _FakeTensorType  # type: ignore[attr-defined]
    tflite_mod.TensorType = tensortype_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tflite", tflite_mod)
    monkeypatch.setitem(sys.modules, "tflite.TensorType", tensortype_mod)


FROM_TENSORFLOW_DTYPE_CASES = [
    (_FakeTensorType.FLOAT16, DataType.FLOAT16),
    (_FakeTensorType.FLOAT32, DataType.FLOAT32),
    (_FakeTensorType.FLOAT64, DataType.FLOAT64),
    (_FakeTensorType.INT16, DataType.INT16),
    (_FakeTensorType.INT32, DataType.INT32),
    (_FakeTensorType.INT64, DataType.INT64),
    (_FakeTensorType.UINT8, DataType.UINT8),
    (_FakeTensorType.UINT16, DataType.UINT16),
    (_FakeTensorType.UINT32, DataType.UINT32),
    (_FakeTensorType.BOOL, DataType.BOOLEAN),
    (_FakeTensorType.STRING, DataType.STRING),
]


@pytest.mark.usefixtures("fake_tflite")
@pytest.mark.parametrize(("dtype", "expected"), FROM_TENSORFLOW_DTYPE_CASES)
def test_from_tensorflow_dtype_supported(dtype: int, expected: DataType):
    assert DataType.from_tensorflow_dtype(dtype) is expected


@pytest.mark.usefixtures("fake_tflite")
def test_from_tensorflow_dtype_unsupported():
    with pytest.raises(ValueError, match="Unsupported TensorFlow"):
        DataType.from_tensorflow_dtype(-1)


AS_DAI_DTYPE_SUPPORTED = [
    DataType.FLOAT16,
    DataType.FLOAT32,
    DataType.FLOAT64,
    DataType.INT8,
    DataType.INT32,
    DataType.UINT8,
    DataType.UINT16,
    DataType.UFXP8,
    DataType.UFXP16,
]


@pytest.mark.parametrize("dtype", AS_DAI_DTYPE_SUPPORTED)
def test_as_dai_dtype_supported(dtype: DataType):
    import depthai as dai

    dai_dtype = dtype.as_dai_dtype()
    assert isinstance(dai_dtype, dai.TensorInfo.DataType)
    assert dtype.supports_dai_dtype()


def test_as_dai_dtype_unsupported():
    assert not DataType.INT64.supports_dai_dtype()
    with pytest.raises(ValueError, match="DepthAI data type"):
        DataType.INT64.as_dai_dtype()


@pytest.mark.parametrize("dtype", list(DataType))
def test_supports_dai_dtype_agrees_with_as_dai_dtype(dtype: DataType):
    """The advertised support set is exactly the convertible set."""
    if dtype.supports_dai_dtype():
        assert dtype.as_dai_dtype() is not None
    else:
        with pytest.raises(ValueError, match="DepthAI data type"):
            dtype.as_dai_dtype()


AS_NUMPY_DTYPE_CASES = [
    (DataType.BFLOAT16, np.float32),
    (DataType.FLOAT16, np.float16),
    (DataType.FLOAT32, np.float32),
    (DataType.FLOAT64, np.float64),
    (DataType.INT4, np.int8),
    (DataType.INT8, np.int8),
    (DataType.INT16, np.int16),
    (DataType.INT32, np.int32),
    (DataType.INT64, np.int64),
    (DataType.UINT4, np.uint8),
    (DataType.UINT8, np.uint8),
    (DataType.UINT16, np.uint16),
    (DataType.UINT32, np.uint32),
    (DataType.UINT64, np.uint64),
    (DataType.BOOLEAN, np.bool_),
    (DataType.STRING, np.str_),
    (DataType.UFXP8, np.uint8),
    (DataType.UFXP16, np.uint16),
    (DataType.UFXP32, np.uint32),
    (DataType.UFXP64, np.uint64),
    (DataType.FXP8, np.int8),
    (DataType.FXP16, np.int16),
    (DataType.FXP32, np.int32),
    (DataType.FXP64, np.int64),
]


@pytest.mark.parametrize(("dtype", "expected"), AS_NUMPY_DTYPE_CASES)
def test_as_numpy_dtype_all(dtype: DataType, expected: type):
    assert dtype.as_numpy_dtype() is expected


@pytest.mark.parametrize(
    ("np_dtype", "dtype"), list(CONVERTERS["numpy"].cases.items())
)
def test_numpy_dtype_round_trips(np_dtype: type, dtype: DataType):
    """The numpy map and its inverse agree on the lossless dtypes."""
    assert dtype.as_numpy_dtype() is np_dtype
    assert DataType.from_numpy_dtype(dtype.as_numpy_dtype()) is dtype


AS_OPENVINO_DTYPE_CASES = [
    (DataType.FLOAT16, "f16"),
    (DataType.FLOAT32, "f32"),
    (DataType.FLOAT64, "f64"),
    (DataType.INT8, "i8"),
    (DataType.INT16, "i16"),
    (DataType.INT32, "i32"),
    (DataType.INT64, "i64"),
    (DataType.UINT8, "u8"),
    (DataType.UINT16, "u16"),
    (DataType.UINT32, "u32"),
    (DataType.UINT64, "u64"),
]


@pytest.mark.parametrize(("dtype", "expected"), AS_OPENVINO_DTYPE_CASES)
def test_as_openvino_dtype_supported(dtype: DataType, expected: str):
    assert dtype.as_openvino_dtype() == expected


def test_as_openvino_dtype_unsupported():
    with pytest.raises(ValueError, match="OpenVINO data type"):
        DataType.BFLOAT16.as_openvino_dtype()


@pytest.mark.parametrize(
    "dtype", [dtype for dtype, _ in AS_OPENVINO_DTYPE_CASES]
)
def test_openvino_dtype_round_trips_through_ir_runtime(dtype: DataType):
    """OpenVINO emits the spelling the IR-runtime reader consumes.

    The two maps are written out independently, so a typo in either one
    would break a real IR conversion; round-tripping ties them together.
    """
    assert DataType.from_ir_runtime_dtype(dtype.as_openvino_dtype()) is dtype


@pytest.mark.parametrize("dtype", list(DataType))
def test_as_snpe_dtype_all(dtype: DataType):
    assert dtype.as_snpe_dtype() == dtype.value


AS_NN_ARCHIVE_DTYPE_CASES = [
    (DataType.UFXP8, "uint8"),
    (DataType.UFXP16, "uint16"),
    (DataType.UFXP32, "uint32"),
    (DataType.UFXP64, "uint64"),
    (DataType.FXP8, "int8"),
    (DataType.FXP16, "int16"),
    (DataType.FXP32, "int32"),
    (DataType.FXP64, "int64"),
    (DataType.FLOAT32, "float32"),
    (DataType.INT8, "int8"),
    (DataType.STRING, "string"),
]


@pytest.mark.parametrize(("dtype", "expected"), AS_NN_ARCHIVE_DTYPE_CASES)
def test_as_nn_archive_dtype_all(dtype: DataType, expected: str):
    assert dtype.as_nn_archive_dtype() == expected


def _width(value: str) -> str:
    """The trailing bit-width digits of a dtype name, if any."""
    match = re.search(r"\d+$", value)
    return match.group() if match else ""


@pytest.mark.parametrize("dtype", list(DataType))
def test_nn_archive_dtype_drops_fixed_point_but_keeps_width(dtype: DataType):
    """NN Archive has no fixed-point dtypes, only same-width integers.

    Whatever the rewrite does, it may not change the bit width or the
    signedness of the dtype it is standing in for.
    """
    archive = dtype.as_nn_archive_dtype()
    assert "fxp" not in archive
    assert re.sub(r"\d+$", "", archive) in {
        "bfloat",
        "float",
        "int",
        "uint",
        "boolean",
        "string",
    }
    assert _width(archive) == _width(dtype.value)
    assert archive.startswith("u") == dtype.value.startswith("u")


INPUT_FILE_TYPE_FROM_PATH_CASES = [
    ("model.onnx", InputFileType.ONNX),
    ("model.xml", InputFileType.IR),
    ("model.bin", InputFileType.IR),
    ("model.tflite", InputFileType.TFLITE),
    ("model.dlc", InputFileType.DLC),
    ("model.har", InputFileType.HAR),
    ("model.pt", InputFileType.PYTORCH),
    ("model.pth", InputFileType.PYTORCH),
]

KNOWN_SUFFIXES = {
    Path(path).suffix for path, _ in INPUT_FILE_TYPE_FROM_PATH_CASES
}


@pytest.mark.parametrize(("path", "expected"), INPUT_FILE_TYPE_FROM_PATH_CASES)
def test_input_file_type_from_path_supported_str(
    path: str, expected: InputFileType
):
    assert InputFileType.from_path(path) is expected


def test_input_file_type_from_path_supported_path_object():
    assert (
        InputFileType.from_path(Path("dir/model.onnx")) is InputFileType.ONNX
    )


def test_input_file_type_from_path_unsupported():
    with pytest.raises(ValueError, match="Unsupported file type"):
        InputFileType.from_path("model.weird")


@given(
    stem=st.from_regex(r"[A-Za-z0-9_.@ -]{1,8}", fullmatch=True),
    suffix=st.sampled_from(sorted(KNOWN_SUFFIXES)),
    parent=st.sampled_from(["", "dir", "a/b", "/abs/dir"]),
)
def test_input_file_type_depends_only_on_the_suffix(
    stem: str, suffix: str, parent: str
):
    """Only the extension decides the format -- never the stem or dirs."""
    assume(Path(f"{stem}{suffix}").suffix == suffix)
    expected = InputFileType.from_path(f"model{suffix}")
    assert (
        InputFileType.from_path(Path(parent) / f"{stem}{suffix}") is expected
    )
