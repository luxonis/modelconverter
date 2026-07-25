"""Host-side unit tests for ``modelconverter.utils.calibration_data``.

The LDF loaders and remote downloads are monkeypatched so no dataset,
network access or cloud credentials are ever required.
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from modelconverter.utils import calibration_data as cd
from modelconverter.utils.constants import CALIBRATION_DIR, SHARED_DIR
from modelconverter.utils.exceptions import ModelconverterException


def _touch_img(directory: Path, name: str = "img.png") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\x89PNG")
    return path


class TestReadImgDir:
    def test_slice_with_max_images(self, work_dir: Path):
        d = work_dir / "imgs"
        d.mkdir()
        for i in range(3):
            (d / f"{i}.png").write_bytes(b"x")
        assert len(cd.read_img_dir(d, max_images=2)) == 2

    def test_all_when_negative(self, work_dir: Path):
        d = work_dir / "imgs"
        d.mkdir()
        for i in range(3):
            (d / f"{i}.png").write_bytes(b"x")
        assert len(cd.read_img_dir(d, max_images=-1)) == 3

    def test_no_images_exits(self, work_dir: Path):
        d = work_dir / "empty"
        d.mkdir()
        with pytest.raises(SystemExit):
            cd.read_img_dir(d, max_images=-1)


class TestFindContentRoot:
    def test_single_nested_recursion(self, work_dir: Path):
        root = work_dir / "root"
        leaf = root / "a" / "b"
        _touch_img(leaf)
        assert cd._find_content_root(root) == leaf

    def test_files_at_top_level_stop(self, work_dir: Path):
        root = work_dir / "root"
        _touch_img(root)
        assert cd._find_content_root(root) == root

    def test_multiple_subdirs_stop(self, work_dir: Path):
        root = work_dir / "root"
        (root / "a").mkdir(parents=True)
        (root / "b").mkdir(parents=True)
        assert cd._find_content_root(root) == root

    def test_ignored_entries(self, work_dir: Path):
        root = work_dir / "root"
        (root / "__MACOSX").mkdir(parents=True)
        (root / ".hidden").write_text("x")
        _touch_img(root / "real")
        # The ignored dir and dotfile are filtered, leaving a single
        # subdir to recurse into.
        assert cd._find_content_root(root) == root / "real"


class TestGetFromRemote:
    def _make_zip(self, work_dir: Path) -> Path:
        src = work_dir / "src"
        _touch_img(src)
        zip_path = CALIBRATION_DIR / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(src / "img.png", arcname="img.png")
        return zip_path

    def test_zip_extracts(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        zip_path = self._make_zip(work_dir)
        monkeypatch.setattr(
            cd,
            "download_from_remote",
            lambda string, dest, max_images=-1: zip_path,
        )
        result = cd._get_from_remote("s3://b/data.zip", CALIBRATION_DIR)
        assert result == CALIBRATION_DIR / "data"
        assert (result / "img.png").exists()

    def test_zip_removes_existing_extract_dir(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        zip_path = self._make_zip(work_dir)
        # Pre-create the extraction target to hit the rmtree branch.
        stale = CALIBRATION_DIR / "data"
        stale.mkdir(parents=True)
        (stale / "stale.txt").write_text("x")
        monkeypatch.setattr(
            cd,
            "download_from_remote",
            lambda string, dest, max_images=-1: zip_path,
        )
        result = cd._get_from_remote("s3://b/data.zip", CALIBRATION_DIR)
        assert result == CALIBRATION_DIR / "data"
        assert not (result / "stale.txt").exists()

    def test_non_zip_passthrough(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        target = work_dir / "plain"
        _touch_img(target)
        monkeypatch.setattr(
            cd,
            "download_from_remote",
            lambda string, dest, max_images=-1: target,
        )
        assert cd._get_from_remote("s3://b/plain", CALIBRATION_DIR) == target


class TestDownloadCalibrationData:
    def test_local_existing_dir(self, work_dir: Path):
        d = work_dir / "calib"
        d.mkdir()
        assert cd.download_calibration_data(str(d)) == Path(str(d))

    def test_shared_dir_fallback(self, work_dir: Path):
        (SHARED_DIR / "calibdir").mkdir(parents=True)
        result = cd.download_calibration_data("calibdir")
        assert result == SHARED_DIR / "calibdir"

    def test_existing_file_raises(self, work_dir: Path):
        f = work_dir / "notadir"
        f.write_text("x")
        with pytest.raises(ModelconverterException, match="not a directory"):
            cd.download_calibration_data(str(f))

    def test_remote_branch(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        target = work_dir / "remote"
        _touch_img(target)
        monkeypatch.setattr(
            cd,
            "download_from_remote",
            lambda string, dest, max_images=-1: target,
        )
        assert cd.download_calibration_data("s3://b/remote") == target

    def test_ldf_two_part(self, monkeypatch: pytest.MonkeyPatch):
        calls: list = []
        monkeypatch.setattr(
            cd,
            "load_from_ldf",
            lambda *a: calls.append(a) or Path("out"),
        )
        assert cd.download_calibration_data("dataset:train") == Path("out")
        # The two-part `match` case calls load_from_ldf with two positional
        # args (loader_plugin defaults inside load_from_ldf).
        assert calls == [("dataset", "train")]

    def test_ldf_three_part(self, monkeypatch: pytest.MonkeyPatch):
        calls: list = []
        monkeypatch.setattr(
            cd,
            "load_from_ldf",
            lambda *a: calls.append(a) or Path("out"),
        )
        assert cd.download_calibration_data("dataset:train:plug") == Path(
            "out"
        )
        assert calls == [("dataset", "train", "plug")]

    def test_ldf_malformed_one_part(self):
        with pytest.raises(ModelconverterException, match="LDF"):
            cd.download_calibration_data("justname")

    def test_ldf_malformed_too_many_parts(self):
        with pytest.raises(ModelconverterException, match="LDF"):
            cd.download_calibration_data("a:b:c:d")


class _FakeLoader:
    def __init__(self, items: list):
        self._items = items

    def __iter__(self):
        return iter(self._items)


class TestLoadFromLdf:
    def test_dataset_loader_writes_images(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        loader = _FakeLoader([(img, None), (img, None)])
        monkeypatch.setattr(
            cd, "LuxonisDataset", lambda name: ("dataset", name)
        )
        monkeypatch.setattr(cd, "LuxonisLoader", lambda dataset, view: loader)
        result = cd.load_from_ldf("myset", "train")
        assert result == CALIBRATION_DIR / "myset"
        assert (result / "0.png").exists()
        assert (result / "1.png").exists()

    def test_multi_input_raises(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        loader = _FakeLoader([({"a": 1}, None)])
        monkeypatch.setattr(cd, "LuxonisDataset", lambda name: None)
        monkeypatch.setattr(cd, "LuxonisLoader", lambda dataset, view: loader)
        with pytest.raises(NotImplementedError):
            cd.load_from_ldf("myset", "train")

    def test_loader_plugin_branch(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        loader = _FakeLoader([(img, None)])

        def fake_plugin(view: str) -> _FakeLoader:
            assert view == "train"
            return loader

        monkeypatch.setattr(cd.LOADERS, "get", lambda plugin: fake_plugin)
        result = cd.load_from_ldf("myset", "train", loader_plugin="Custom")
        assert result == CALIBRATION_DIR / "myset"
        assert (result / "0.png").exists()
