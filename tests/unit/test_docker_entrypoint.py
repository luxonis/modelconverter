"""Tests for the container entrypoint.

The signal and stdin handling lives in ``docker/entrypoint_common.sh``,
which every target entrypoint sources; running the RVC2 one therefore
exercises the logic of all of them. The other targets add environment
setup that only exists inside their image (SNPE, Hailo), so they cannot
be run here.

Its ownership handling runs against ``APP_ROOT``, which the image leaves
at ``/app``; the tests point it at a temp tree and put a recording
``chown`` on PATH, since letting a real ``chown -R`` run as part of the
suite is not an option.
"""

import os
import shutil
import signal
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

DOCKER_DIR = Path(__file__).parents[2] / "docker"

# The entrypoint is a bash script with a `#!/bin/bash` shebang. That path is
# guaranteed inside the image but not on every developer machine, so the tests
# invoke it through whichever bash is on PATH instead of relying on the shebang.
_BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(_BASH is None, reason="bash is not available")

BASH = _BASH or "bash"

# An entrypoint that stops forwarding a signal or handing over its stdin must
# fail the test run rather than block it.
TIMEOUT = 5


@pytest.fixture
def entrypoint(tmp_path: Path) -> Path:
    """Lays the entrypoint out the way the image does, with the shared
    part next to it."""
    app = tmp_path / "app"
    app.mkdir()
    shutil.copy(DOCKER_DIR / "rvc2" / "entrypoint.sh", app / "entrypoint.sh")
    shutil.copy(
        DOCKER_DIR / "entrypoint_common.sh", app / "entrypoint_common.sh"
    )
    return app / "entrypoint.sh"


def _entrypoint_env(tmp_path: Path, body: str) -> dict[str, str]:
    """Puts a fake ``modelconverter`` on PATH and returns the
    environment to run the entrypoint with."""
    executable = tmp_path / "modelconverter"
    executable.write_text(f"#!/usr/bin/env bash\n{body}")
    executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    # A developer who followed the docker-compose instructions has these
    # exported; the entrypoint would then try to chown paths that only exist
    # inside the image.
    env.pop("HOST_UID", None)
    env.pop("HOST_GID", None)
    return env


@contextmanager
def _entrypoint_process(
    entrypoint: Path,
    *args: str,
    env: dict[str, str],
    stdin: bool = False,
) -> Generator[subprocess.Popen, None, None]:
    """Runs the entrypoint and leaves nothing of it behind.

    The entrypoint runs the converter as a background child, so killing
    the entrypoint alone -- all a timeout does -- would leave that child
    spinning for the rest of the test session. Its own session makes the
    pair killable as one group.
    """
    process = subprocess.Popen(
        [BASH, str(entrypoint), *args],
        env=env,
        text=True,
        stdin=subprocess.PIPE if stdin else None,
        start_new_session=True,
    )
    try:
        yield process
    finally:
        # Only while the process is unreaped, so the pid cannot have been
        # recycled into an unrelated group by the time it is signalled.
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _wait_for(path: Path, timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return False


def test_entrypoint_gives_the_command_a_real_stdin(
    tmp_path: Path, entrypoint: Path
) -> None:
    env = _entrypoint_env(tmp_path, 'read -r line\nexit "$line"\n')

    with _entrypoint_process(
        entrypoint, "convert", "rvc2", env=env, stdin=True
    ) as process:
        process.communicate("7\n", timeout=TIMEOUT)

    assert process.returncode == 7


@pytest.mark.skipif(
    not Path("/bin/bash").exists(), reason="/bin/bash is not available"
)
def test_entrypoint_keeps_interactive_shell_stdin(
    tmp_path: Path, entrypoint: Path
) -> None:
    with _entrypoint_process(
        entrypoint, env=_entrypoint_env(tmp_path, "exit 1\n"), stdin=True
    ) as process:
        process.communicate("exit 7\n", timeout=TIMEOUT)

    assert process.returncode == 7


@pytest.mark.parametrize(
    ("sent", "exit_code"), [(signal.SIGTERM, 42), (signal.SIGINT, 130)]
)
def test_entrypoint_forwards_signals_and_waits_for_child_cleanup(
    tmp_path: Path, entrypoint: Path, sent: signal.Signals, exit_code: int
) -> None:
    signal_name = sent.name.removeprefix("SIG")
    signal_file = tmp_path / "signal"
    ready_file = tmp_path / "ready"
    env = _entrypoint_env(
        tmp_path,
        f"trap 'printf {signal_name} > \"$SIGNAL_FILE\"; exit {exit_code}' "
        f"{signal_name}\n"
        'printf ready > "$READY_FILE"\n'
        "while true; do sleep 0.05; done\n",
    )
    env |= {"READY_FILE": str(ready_file), "SIGNAL_FILE": str(signal_file)}

    with _entrypoint_process(
        entrypoint, "convert", "rvc2", env=env
    ) as process:
        assert _wait_for(ready_file)

        process.send_signal(sent)
        assert process.wait(timeout=TIMEOUT) == exit_code
        assert signal_file.read_text() == signal_name


@pytest.fixture
def mounts(tmp_path: Path) -> Path:
    """An ``APP_ROOT`` laid out the way the container's mounts are."""
    app = tmp_path / "app_root"
    (app / "output").mkdir(parents=True)
    cache = app / "shared_with_container"
    (cache / "inputs" / "digest").mkdir(parents=True)
    (cache / "models").mkdir()
    return app


def _recording_chown(tmp_path: Path) -> Path:
    """Puts a ``chown`` that only records its arguments on PATH."""
    recorder = tmp_path / "chown"
    recorder.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CHOWN_LOG"\n'
    )
    recorder.chmod(0o755)
    return tmp_path / "chown.log"


def _handover_env(tmp_path: Path, mounts: Path, log: Path) -> dict[str, str]:
    return _entrypoint_env(tmp_path, "exit 0\n") | {
        "HOST_UID": "1234",
        "HOST_GID": "5678",
        "APP_ROOT": str(mounts),
        "CHOWN_LOG": str(log),
    }


def _run_handover(entrypoint: Path, env: dict[str, str]) -> None:
    with _entrypoint_process(
        entrypoint, "convert", "rvc2", env=env
    ) as process:
        assert process.wait(timeout=TIMEOUT) == 0


def test_entrypoint_hands_the_mounts_back_to_the_host_user(
    tmp_path: Path, entrypoint: Path, mounts: Path
) -> None:
    log = _recording_chown(tmp_path)

    _run_handover(entrypoint, _handover_env(tmp_path, mounts, log))

    cache = mounts / "shared_with_container"
    chowned = log.read_text().splitlines()
    assert f"-R 1234:5678 {mounts / 'output'}" in chowned
    # The cache root itself. When its host directory did not exist the daemon
    # created it as root, and nothing else ever repairs that.
    assert f"1234:5678 {cache}" in chowned
    assert f"-R 1234:5678 {cache / 'models'}" in chowned
    # The staged inputs are written by the host user and only read in the
    # container, so walking them after every run would be pure cost.
    assert not any("inputs" in line for line in chowned)


def test_entrypoint_hands_back_the_dev_mounts_it_finds(
    tmp_path: Path, entrypoint: Path, mounts: Path
) -> None:
    """A dev container mounts the host checkout over the installed
    package and the suite over the image's, and running as root leaves
    ``__pycache__`` in both."""
    (mounts / "tests").mkdir()
    (mounts / "modelconverter").mkdir()
    log = _recording_chown(tmp_path)

    _run_handover(entrypoint, _handover_env(tmp_path, mounts, log))

    chowned = log.read_text().splitlines()
    assert f"-R 1234:5678 {mounts / 'tests'}" in chowned
    assert f"-R 1234:5678 {mounts / 'modelconverter'}" in chowned


def test_entrypoint_leaves_ownership_alone_without_a_host_identity(
    tmp_path: Path, entrypoint: Path, mounts: Path
) -> None:
    """Under a rootless or user-namespaced daemon the launcher passes no
    identity, and chowning anyway would hand the files to an unmapped
    sub-uid."""
    log = _recording_chown(tmp_path)
    env = _handover_env(tmp_path, mounts, log)
    del env["HOST_UID"]

    _run_handover(entrypoint, env)

    assert not log.exists()
