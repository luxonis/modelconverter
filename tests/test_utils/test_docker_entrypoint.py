"""Tests for the container entrypoint.

The signal, stdin and ownership handling lives in
``docker/entrypoint_common.sh``, which every target entrypoint sources;
running the RVC2 one therefore exercises the logic of all of them. The
other targets add environment setup that only exists inside their image
(SNPE, Hailo), so they cannot be run here.
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

DOCKER_DIR = Path(__file__).parents[2] / "docker"

# The entrypoint is a bash script with a `#!/bin/bash` shebang. That path is
# guaranteed inside the image but not on every developer machine, so the tests
# invoke it through whichever bash is on PATH instead of relying on the shebang.
_BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(_BASH is None, reason="bash is not available")

BASH = _BASH or "bash"


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

    result = subprocess.run(
        [BASH, str(entrypoint), "convert", "rvc2"],
        input="7\n",
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 7


@pytest.mark.skipif(
    not Path("/bin/bash").exists(), reason="/bin/bash is not available"
)
def test_entrypoint_keeps_interactive_shell_stdin(
    tmp_path: Path, entrypoint: Path
) -> None:
    result = subprocess.run(
        [BASH, str(entrypoint)],
        input="exit 7\n",
        text=True,
        env=_entrypoint_env(tmp_path, "exit 1\n"),
        check=False,
    )

    assert result.returncode == 7


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

    process = subprocess.Popen(
        [BASH, str(entrypoint), "convert", "rvc2"],
        env=env,
    )
    try:
        assert _wait_for(ready_file)

        process.send_signal(sent)
        assert process.wait(timeout=5) == exit_code
        assert signal_file.read_text() == signal_name
    finally:
        if process.poll() is None:
            process.kill()
