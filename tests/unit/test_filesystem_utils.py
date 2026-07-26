"""Host-side unit tests for ``modelconverter.utils.filesystem_utils``.

All remote I/O is monkeypatched at the ``LuxonisFileSystem`` boundary so
the suite never touches the network or any cloud credentials.
"""

from pathlib import Path

import pytest

from modelconverter.utils import filesystem_utils as fsu
from modelconverter.utils.constants import SHARED_DIR


def _install_fake_fs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    is_dir: bool = False,
    walk: tuple[str, ...] = (),
    split: tuple[str, str] | None = None,
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
        def split_full_path(url: str) -> tuple[str, str]:  # pragma: no cover
            if split is not None:
                return split
            base, _, name = url.rpartition("/")
            return base, name

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


class TestGetProtocol:
    def test_file(self):
        assert fsu.get_protocol("/data/model.onnx") == "file"

    def test_s3(self):
        assert fsu.get_protocol("s3://bucket/model.onnx") == "s3"

    def test_gs(self):
        assert fsu.get_protocol("gs://bucket/model.onnx") == "gcs"


class TestResolvePath:
    def test_existing_absolute_file(self, work_dir: Path):
        f = work_dir / "model.onnx"
        f.write_text("x")
        assert fsu.resolve_path(str(f), work_dir) == f

    def test_shared_dir_fallback(self, work_dir: Path):
        target = SHARED_DIR / "misc" / "cfg.yaml"
        target.write_text("x")
        # ``misc/cfg.yaml`` does not exist under cwd, but does under
        # ``shared_with_container/``.
        resolved = fsu.resolve_path("misc/cfg.yaml", work_dir)
        assert resolved == SHARED_DIR / "misc" / "cfg.yaml"

    def test_missing_raises(self, work_dir: Path):
        with pytest.raises(ValueError, match="does not exist"):
            fsu.resolve_path("nope.onnx", work_dir)

    def test_remote_branch(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
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


class TestDownloadFromRemote:
    def test_single_file(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        instances = _install_fake_fs(
            monkeypatch, is_dir=False, split=("s3://b", "model.onnx")
        )
        result = fsu.download_from_remote("s3://b/model.onnx", work_dir)
        expected = work_dir / "model.onnx"
        assert result == expected
        assert instances[0].get_file_calls == [("model.onnx", str(expected))]

    def test_dest_as_str(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _install_fake_fs(
            monkeypatch, is_dir=False, split=("s3://b", "model.onnx")
        )
        result = fsu.download_from_remote("s3://b/model.onnx", str(work_dir))
        assert result == work_dir / "model.onnx"

    def test_single_file_cached_skips(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # local_path exists and the unsafe-cache flag is set -> no fetch.
        (work_dir / "model.onnx").write_text("cached")
        monkeypatch.setenv("MODELCONVERTER_UNSAFE_CACHE", "1")
        instances = _install_fake_fs(
            monkeypatch, is_dir=False, split=("s3://b", "model.onnx")
        )
        result = fsu.download_from_remote("s3://b/model.onnx", work_dir)
        assert result == work_dir / "model.onnx"
        assert instances[0].get_file_calls == []

    def test_directory_with_max_files_cutoff(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("MODELCONVERTER_UNSAFE_CACHE", raising=False)
        walk = ("d/f0.jpg", "d/f1.jpg", "d/f2.jpg")
        instances = _install_fake_fs(
            monkeypatch, is_dir=True, walk=walk, split=("s3://b", "d")
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

    def test_directory_cached_skips(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (work_dir / "d").mkdir()
        monkeypatch.setenv("MODELCONVERTER_UNSAFE_CACHE", "1")
        instances = _install_fake_fs(
            monkeypatch,
            is_dir=True,
            walk=("d/f0.jpg", "d/f1.jpg"),
            split=("s3://b", "d"),
        )
        result = fsu.download_from_remote("s3://b/d", work_dir)
        assert result == work_dir / "d"
        assert instances[0].get_file_calls == []


class TestUploadToRemote:
    def test_upload_file(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        local = work_dir / "model.onnx"
        local.write_text("x")
        instances = _install_fake_fs(
            monkeypatch, split=("s3://b", "model.onnx")
        )
        fsu.upload_to_remote(
            str(local), "s3://b/model.onnx", put_file_plugin="plug"
        )
        fs = instances[0]
        assert fs.put_file_plugin == "plug"
        assert fs.put_file_calls == [(local, "model.onnx")]
        assert fs.put_dir_calls == []

    def test_upload_dir(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        local = work_dir / "d"
        local.mkdir()
        instances = _install_fake_fs(monkeypatch, split=("s3://b", "d"))
        fsu.upload_to_remote(local, "s3://b/d")
        fs = instances[0]
        assert fs.put_dir_calls == [(local, "d")]
        assert fs.put_file_calls == []
