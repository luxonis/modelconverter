"""Host-side unit tests for ``modelconverter.utils.filesystem_utils``.

All remote I/O is monkeypatched at the ``LuxonisFileSystem`` boundary so
the suite never touches the network or any cloud credentials.
"""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from modelconverter.utils import filesystem_utils as fsu
from modelconverter.utils.constants import SHARED_DIR
from tests.helpers.strategies import reuses_function_fixtures


def _install_fake_fs(
    monkeypatch: pytest.MonkeyPatch,
    split: tuple[str, str],
    *,
    is_dir: bool = False,
    walk: tuple[str, ...] = (),
) -> list:
    """Patch ``filesystem_utils.LuxonisFileSystem`` with a fake.

    Returns a list that collects every constructed fake instance so the
    tests can assert on recorded calls.
    """
    instances: list = []

    class FakeFS:
        def __init__(
            self, absolute_path: str, put_file_plugin: str | None = None
        ) -> None:
            self.absolute_path = absolute_path
            self.put_file_plugin = put_file_plugin
            self.get_file_calls: list = []
            self.put_file_calls: list = []
            self.put_dir_calls: list = []
            instances.append(self)

        @staticmethod
        def split_full_path(url: str) -> tuple[str, str]:
            return split

        def is_directory(self, remote_path: str) -> bool:
            return is_dir

        def walk_dir(self, remote_path: str) -> list:
            return list(walk)

        def get_file(self, remote: str, local: str) -> None:
            self.get_file_calls.append((remote, local))

        def put_file(self, local: str | Path, remote: str) -> None:
            self.put_file_calls.append((local, remote))

        def put_dir(self, local: str | Path, remote: str) -> None:
            self.put_dir_calls.append((local, remote))

    monkeypatch.setattr(fsu, "LuxonisFileSystem", FakeFS)
    return instances


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("/data/model.onnx", "file"),
        ("s3://bucket/model.onnx", "s3"),
        ("gs://bucket/model.onnx", "gcs"),
    ],
)
def test_get_protocol(url: str, expected: str):
    assert fsu.get_protocol(url) == expected


def test_resolve_path_keeps_an_existing_absolute_path(work_dir: Path):
    f = work_dir / "model.onnx"
    f.write_text("x")
    assert fsu.resolve_path(str(f), work_dir) == f


def test_resolve_path_falls_back_to_the_shared_dir(work_dir: Path):
    target = SHARED_DIR / "misc" / "cfg.yaml"
    target.write_text("x")
    # ``misc/cfg.yaml`` does not exist under cwd, but does under
    # ``shared_with_container/``.
    resolved = fsu.resolve_path("misc/cfg.yaml", work_dir)
    assert resolved == SHARED_DIR / "misc" / "cfg.yaml"


def test_resolve_path_missing_raises(work_dir: Path):
    with pytest.raises(ValueError, match="does not exist"):
        fsu.resolve_path("nope.onnx", work_dir)


def test_resolve_path_downloads_a_remote_url(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    downloaded = work_dir / "downloaded.onnx"
    downloaded.write_text("x")
    monkeypatch.setattr(
        fsu,
        "download_from_remote",
        lambda url, dest: downloaded,
    )
    resolved = fsu.resolve_path("s3://bucket/downloaded.onnx", work_dir)
    assert resolved == downloaded


def test_download_single_file(work_dir: Path, monkeypatch: pytest.MonkeyPatch):
    instances = _install_fake_fs(monkeypatch, ("s3://b", "model.onnx"))
    result = fsu.download_from_remote("s3://b/model.onnx", work_dir)
    expected = work_dir / "model.onnx"
    assert result == expected
    assert instances[0].get_file_calls == [("model.onnx", str(expected))]


def test_download_accepts_a_str_destination(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_fs(monkeypatch, ("s3://b", "model.onnx"))
    result = fsu.download_from_remote("s3://b/model.onnx", str(work_dir))
    assert result == work_dir / "model.onnx"


def test_download_single_file_uses_the_cache(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    # local_path exists and the unsafe-cache flag is set -> no fetch.
    (work_dir / "model.onnx").write_text("cached")
    monkeypatch.setenv("MODELCONVERTER_UNSAFE_CACHE", "1")
    instances = _install_fake_fs(monkeypatch, ("s3://b", "model.onnx"))
    result = fsu.download_from_remote("s3://b/model.onnx", work_dir)
    assert result == work_dir / "model.onnx"
    assert instances[0].get_file_calls == []


def test_directory_with_max_files_cutoff(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("MODELCONVERTER_UNSAFE_CACHE", raising=False)
    walk = ("d/f0.jpg", "d/f1.jpg", "d/f2.jpg")
    instances = _install_fake_fs(
        monkeypatch, ("s3://b", "d"), is_dir=True, walk=walk
    )
    result = fsu.download_from_remote("s3://b/d", work_dir, max_files=2)
    local_path = work_dir / "d"
    assert result == local_path
    # Only the first two files fetched before the ``i == max_files``
    # break.
    assert instances[0].get_file_calls == [
        ("d/f0.jpg", str(local_path / "f0.jpg")),
        ("d/f1.jpg", str(local_path / "f1.jpg")),
    ]


def test_download_directory_uses_the_cache(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    (work_dir / "d").mkdir()
    monkeypatch.setenv("MODELCONVERTER_UNSAFE_CACHE", "1")
    instances = _install_fake_fs(
        monkeypatch,
        ("s3://b", "d"),
        is_dir=True,
        walk=("d/f0.jpg", "d/f1.jpg"),
    )
    result = fsu.download_from_remote("s3://b/d", work_dir)
    assert result == work_dir / "d"
    assert instances[0].get_file_calls == []


def test_upload_file_to_remote(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    local = work_dir / "model.onnx"
    local.write_text("x")
    instances = _install_fake_fs(monkeypatch, ("s3://b", "model.onnx"))
    fsu.upload_to_remote(
        str(local), "s3://b/model.onnx", put_file_plugin="plug"
    )
    fs = instances[0]
    assert fs.put_file_plugin == "plug"
    assert fs.put_file_calls == [(local, "model.onnx")]
    assert fs.put_dir_calls == []


def test_upload_dir_to_remote(work_dir: Path, monkeypatch: pytest.MonkeyPatch):
    local = work_dir / "d"
    local.mkdir()
    instances = _install_fake_fs(monkeypatch, ("s3://b", "d"))
    fsu.upload_to_remote(local, "s3://b/d")
    fs = instances[0]
    assert fs.put_dir_calls == [(local, "d")]
    assert fs.put_file_calls == []


@reuses_function_fixtures
@given(
    available=st.integers(min_value=0, max_value=8),
    max_files=st.integers(min_value=-1, max_value=10),
)
def test_directory_download_respects_max_files(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    available: int,
    max_files: int,
):
    """``max_files`` caps the fetch; a negative value means "all".

    Calibration downloads lean on this to pull a handful of images out
    of a bucket holding thousands, so both the cutoff and the
    unlimited case have to be exact.
    """
    monkeypatch.delenv("MODELCONVERTER_UNSAFE_CACHE", raising=False)
    walk = tuple(f"d/f{i}.jpg" for i in range(available))
    instances = _install_fake_fs(
        monkeypatch, ("s3://b", "d"), is_dir=True, walk=walk
    )

    fsu.download_from_remote("s3://b/d", work_dir, max_files=max_files)

    expected = available if max_files < 0 else min(max_files, available)
    assert [remote for remote, _ in instances[0].get_file_calls] == list(
        walk[:expected]
    )
