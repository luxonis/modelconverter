"""Tests for the host-side rootless conversion workflow."""

import subprocess
from pathlib import Path
from typing import Any, NamedTuple, cast

import pytest
import yaml

from modelconverter.__main__ import app
from modelconverter.utils import constants, docker_utils
from modelconverter.utils.constants import CONTAINER_SHARED_DIR
from tests.helpers.onnx_factory import single_io_onnx

_IMAGE = "luxonis/modelconverter-rvc4:test"


class _Launch(NamedTuple):
    argv: list[str]
    compose: dict[str, Any]

    @property
    def container_args(self) -> list[str]:
        return self.argv[self.argv.index("modelconverter") + 1 :]

    @property
    def service(self) -> dict[str, Any]:
        return next(iter(self.compose["services"].values()))


def _launch(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    namespace_mode: str = "rootless",
    extra: list[str] | None = None,
) -> _Launch:
    launch: _Launch | None = None

    monkeypatch.setattr(
        docker_utils, "get_docker_image", lambda *_a, **_k: _IMAGE
    )
    monkeypatch.setattr(docker_utils, "docker_bin", lambda: "docker")
    monkeypatch.setattr(
        docker_utils, "docker_user_namespace_mode", lambda: namespace_mode
    )

    def fake_run(
        argv: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess:
        nonlocal launch
        compose_file = Path(argv[argv.index("-f") + 1])
        compose = cast(
            dict[str, Any], yaml.safe_load(compose_file.read_text())
        )
        launch = _Launch(argv, compose)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(docker_utils.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exit_info:
        app.meta(["convert", "rvc4", "--path", str(config), *(extra or [])])
    assert exit_info.value.code == 0
    assert launch is not None

    return launch


def _staged_on_host(container_path: str) -> Path:
    relative = Path(container_path).relative_to(CONTAINER_SHARED_DIR)
    return constants.get_cache_dir() / relative


def _project(work_dir: Path) -> Path:
    project = work_dir / "my project"
    calibration = project / "calibration"
    calibration.mkdir(parents=True)
    (calibration / "0.png").write_bytes(b"calibration image")
    single_io_onnx(project / "model.onnx")

    config = project / "config.yaml"
    config.write_text(
        cast(
            str,
            yaml.safe_dump(
                {
                    "input_model": "model.onnx",
                    "calibration": {"path": "calibration"},
                }
            ),
        )
    )
    return config


def test_config_and_its_references_reach_the_container(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _project(work_dir)

    launch = _launch(config, monkeypatch)

    path_argument = launch.container_args[
        launch.container_args.index("--path") + 1
    ]
    input_prefix = f"{CONTAINER_SHARED_DIR}/inputs/"
    assert path_argument.startswith(input_prefix)

    staged = cast(
        dict[str, Any],
        yaml.safe_load(_staged_on_host(path_argument).read_text()),
    )
    model_path = staged["input_model"]
    calibration_path = staged["calibration"]["path"]
    assert model_path.startswith(input_prefix)
    assert calibration_path.startswith(input_prefix)
    assert _staged_on_host(model_path).is_file()
    assert (_staged_on_host(calibration_path) / "0.png").read_bytes() == (
        b"calibration image"
    )


def test_working_directory_needs_no_shared_with_container(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _project(work_dir)

    _launch(config, monkeypatch)

    assert not (work_dir / "shared_with_container").exists()


def test_only_the_cache_and_output_are_mounted(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _project(work_dir)

    launch = _launch(config, monkeypatch)

    assert launch.service["volumes"] == [
        f"{constants.get_cache_dir()}:{CONTAINER_SHARED_DIR}",
        f"{work_dir / 'output'}:/app/output",
    ]


def test_output_directory_is_created_by_the_host_user(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _project(work_dir)
    output_dir = work_dir / "output"
    output_dir.rmdir()

    _launch(config, monkeypatch)

    assert output_dir.is_dir()


@pytest.mark.parametrize(
    ("namespace_mode", "identity_passed"),
    [
        ("rootful", True),
        ("rootless", False),
        ("userns", False),
    ],
)
def test_host_identity_follows_the_daemon(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    namespace_mode: str,
    identity_passed: bool,
):
    config = _project(work_dir)

    launch = _launch(
        config,
        monkeypatch,
        namespace_mode=namespace_mode,
    )

    environment = launch.service["environment"]
    assert ("HOST_UID" in environment) is identity_passed
    assert ("HOST_GID" in environment) is identity_passed


def test_overrides_reach_the_container_verbatim(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """`rvc4.encodings` takes the encodings JSON inline.

    The entrypoints exec their argv directly, so nothing may rewrite the
    double quotes on the way -- `json.loads` rejects the single-quoted
    result and the conversion never starts.
    """
    config = _project(work_dir)
    encodings = '{"activation_encodings": {"x": [{"bitwidth": 8}]}}'

    launch = _launch(config, monkeypatch, extra=["rvc4.encodings", encodings])

    assert encodings in launch.container_args