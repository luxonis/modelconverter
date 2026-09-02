"""Host-side unit tests for ``modelconverter.utils.calibration_data``.

The LDF loaders and remote downloads are monkeypatched so no dataset,
network access or cloud credentials are ever required.
"""

import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from modelconverter.utils import calibration_data as cd
from modelconverter.utils import filesystem_utils
from modelconverter.utils.constants import CALIBRATION_DIR, SHARED_DIR
from modelconverter.utils.exceptions import ModelconverterException
from tests.helpers.strategies import reuses_function_fixtures


def _touch_img(directory: Path, name: str = "img.png") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\x89PNG")
    return path


def _img_dir(root: Path, count: int) -> Path:
    """Create a directory holding ``count`` (empty) calibration images."""
    directory = root / "imgs"
    directory.mkdir(exist_ok=True)
    for path in directory.iterdir():
        path.unlink()
    for i in range(count):
        (directory / f"{i}.png").write_bytes(b"x")
    return directory


@pytest.fixture
def fake_download(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Path], None]:
    """Make ``download_from_remote`` yield a prepared local path."""

    def install(target: Path) -> None:
        monkeypatch.setattr(
            cd,
            "download_from_remote",
            lambda string, dest, max_images=-1: target,
        )

    return install


@reuses_function_fixtures
@given(
    available=st.integers(min_value=1, max_value=8),
    max_images=st.integers(min_value=-1, max_value=10),
)
def test_read_img_dir_respects_max_images(
    work_dir: Path, available: int, max_images: int
):
    """``max_images`` truncates the listing; negative means "all"."""
    directory = _img_dir(work_dir, available)
    expected = available if max_images < 0 else min(max_images, available)
    assert len(cd.read_img_dir(directory, max_images)) == expected


def test_read_img_dir_with_no_images_exits(work_dir: Path):
    d = work_dir / "empty"
    d.mkdir()
    with pytest.raises(SystemExit):
        cd.read_img_dir(d, max_images=-1)


def test_content_root_descends_through_single_subdirs(work_dir: Path):
    root = work_dir / "root"
    leaf = root / "a" / "b"
    _touch_img(leaf)
    assert cd._find_content_root(root) == leaf


def test_content_root_stops_at_top_level_files(work_dir: Path):
    root = work_dir / "root"
    _touch_img(root)
    assert cd._find_content_root(root) == root


def test_content_root_stops_at_a_fork(work_dir: Path):
    root = work_dir / "root"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    assert cd._find_content_root(root) == root


def test_content_root_skips_dotfiles_and_macosx(work_dir: Path):
    root = work_dir / "root"
    (root / "__MACOSX").mkdir(parents=True)
    (root / ".hidden").write_text("x")
    _touch_img(root / "real")
    # The ignored dir and dotfile are filtered, leaving a single
    # subdir to recurse into.
    assert cd._find_content_root(root) == root / "real"


def _make_zip(work_dir: Path) -> Path:
    src = work_dir / "src"
    _touch_img(src)
    zip_path = CALIBRATION_DIR / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(src / "img.png", arcname="img.png")
    return zip_path


def test_remote_zip_is_extracted(
    work_dir: Path, fake_download: Callable[[Path], None]
):
    fake_download(_make_zip(work_dir))
    result = cd._get_from_remote("s3://b/data.zip", CALIBRATION_DIR)
    assert result == CALIBRATION_DIR / "data"
    assert (result / "img.png").exists()


def test_remote_zip_replaces_a_stale_extract_dir(
    work_dir: Path, fake_download: Callable[[Path], None]
):
    fake_download(_make_zip(work_dir))
    # Pre-create the extraction target to hit the rmtree branch.
    stale = CALIBRATION_DIR / "data"
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_text("x")
    result = cd._get_from_remote("s3://b/data.zip", CALIBRATION_DIR)
    assert result == CALIBRATION_DIR / "data"
    assert not (result / "stale.txt").exists()


def test_remote_dir_is_used_as_is(
    work_dir: Path, fake_download: Callable[[Path], None]
):
    target = work_dir / "plain"
    _touch_img(target)
    fake_download(target)
    assert cd._get_from_remote("s3://b/plain", CALIBRATION_DIR) == target


def test_local_dir_is_used_as_is(work_dir: Path):
    d = work_dir / "calib"
    d.mkdir()
    assert cd.download_calibration_data(str(d)) == Path(str(d))


def test_calibration_dir_falls_back_to_the_shared_dir(
    monkeypatch: pytest.MonkeyPatch,
):
    # The shared directory is only the fallback root inside the container;
    # a native run falls back to the working directory instead.
    monkeypatch.setattr(filesystem_utils, "in_docker", lambda: True)
    (SHARED_DIR / "calibdir").mkdir(parents=True)
    result = cd.download_calibration_data("calibdir")
    assert result == SHARED_DIR / "calibdir"


def test_calibration_path_must_be_a_directory(work_dir: Path):
    f = work_dir / "notadir"
    f.write_text("x")
    with pytest.raises(ModelconverterException, match="not a directory"):
        cd.download_calibration_data(str(f))


def test_calibration_data_downloaded_from_remote(
    work_dir: Path, fake_download: Callable[[Path], None]
):
    target = work_dir / "remote"
    _touch_img(target)
    fake_download(target)
    assert cd.download_calibration_data("s3://b/remote") == target


@pytest.fixture
def recorded_ldf(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace ``load_from_ldf``, recording the arguments it is given."""
    calls: list = []

    def record(*args) -> Path:
        calls.append(args)
        return Path("out")

    monkeypatch.setattr(cd, "load_from_ldf", record)
    return calls


def test_ldf_two_part(recorded_ldf: list):
    assert cd.download_calibration_data("dataset:train") == Path("out")
    # The two-part `match` case calls load_from_ldf with two positional
    # args (loader_plugin defaults inside load_from_ldf).
    assert recorded_ldf == [("dataset", "train")]


def test_ldf_three_part(recorded_ldf: list):
    assert cd.download_calibration_data("dataset:train:plug") == Path("out")
    assert recorded_ldf == [("dataset", "train", "plug")]


def test_ldf_malformed_one_part():
    with pytest.raises(ModelconverterException, match="LDF"):
        cd.download_calibration_data("justname")


def test_ldf_malformed_too_many_parts():
    with pytest.raises(ModelconverterException, match="LDF"):
        cd.download_calibration_data("a:b:c:d")


class _FakeLoader:
    def __init__(self, items: list):
        self._items = items

    def __iter__(self):
        return iter(self._items)


def test_dataset_loader_writes_images(monkeypatch: pytest.MonkeyPatch):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    loader = _FakeLoader([(img, None), (img, None)])
    monkeypatch.setattr(cd, "LuxonisDataset", lambda name: ("dataset", name))
    monkeypatch.setattr(cd, "LuxonisLoader", lambda dataset, view: loader)
    result = cd.load_from_ldf("myset", "train")
    assert result == CALIBRATION_DIR / "myset"
    assert (result / "0.png").exists()
    assert (result / "1.png").exists()


def test_multi_input_dataset_raises(monkeypatch: pytest.MonkeyPatch):
    loader = _FakeLoader([({"a": 1}, None)])
    monkeypatch.setattr(cd, "LuxonisDataset", lambda name: None)
    monkeypatch.setattr(cd, "LuxonisLoader", lambda dataset, view: loader)
    with pytest.raises(NotImplementedError):
        cd.load_from_ldf("myset", "train")


def test_loader_plugin_replaces_the_default_loader(
    monkeypatch: pytest.MonkeyPatch,
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
