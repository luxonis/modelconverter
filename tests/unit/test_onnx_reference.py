"""Tests for ONNX reference-session compatibility fallback."""

from pathlib import Path

import onnx
import onnxruntime as ort
import pytest

from tests.helpers.onnx_factory import single_io_onnx
from tests.helpers.onnx_reference import _create_onnx_session


def test_reference_session_prefers_loading_from_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "large-external-model.onnx"
    expected_session = object()
    calls: list[str | bytes] = []

    def create_session(model: str | bytes, *, providers: list[str]) -> object:
        calls.append(model)
        assert providers == ["CPUExecutionProvider"]
        return expected_session

    monkeypatch.setattr(ort, "InferenceSession", create_session)
    monkeypatch.setattr(
        onnx,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "a path-loadable model must not be deserialized"
        ),
    )

    session = _create_onnx_session(model_path)

    assert session is expected_session
    assert calls == [str(model_path)]


def test_reference_session_lowers_ir_version_after_runtime_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = single_io_onnx(tmp_path / "new-ir.onnx")
    model = onnx.load(model_path)
    model.ir_version = 10
    onnx.save(model, model_path)
    expected_session = object()
    calls: list[str | bytes] = []

    def create_session(source: str | bytes, *, providers: list[str]) -> object:
        calls.append(source)
        assert providers == ["CPUExecutionProvider"]
        if len(calls) == 1:
            raise RuntimeError(
                "Unsupported model IR version: 10, max supported IR version: 8"
            )
        assert isinstance(source, bytes)
        assert onnx.load_model_from_string(source).ir_version == 8
        return expected_session

    monkeypatch.setattr(ort, "InferenceSession", create_session)

    session = _create_onnx_session(model_path)

    assert session is expected_session
    assert calls[0] == str(model_path)
    assert isinstance(calls[1], bytes)


def test_reference_session_does_not_mask_other_runtime_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "invalid.onnx"

    def reject_session(
        _source: str | bytes, *, providers: list[str]
    ) -> object:
        assert providers == ["CPUExecutionProvider"]
        raise RuntimeError("external data file is missing")

    monkeypatch.setattr(ort, "InferenceSession", reject_session)
    monkeypatch.setattr(
        onnx,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "non-IR failures must not deserialize the model"
        ),
    )

    with pytest.raises(RuntimeError, match="external data file is missing"):
        _create_onnx_session(model_path)
