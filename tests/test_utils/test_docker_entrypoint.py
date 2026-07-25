import os
import subprocess
import time
from pathlib import Path


def test_entrypoint_keeps_interactive_shell_stdin() -> None:
    result = subprocess.run(
        ["docker/rvc2/entrypoint.sh"],
        input="exit 7\n",
        text=True,
        check=False,
    )

    assert result.returncode == 7


def test_entrypoint_forwards_sigterm_and_waits_for_child_cleanup(
    tmp_path: Path,
) -> None:
    signal_file = tmp_path / "signal"
    ready_file = tmp_path / "ready"
    executable = tmp_path / "modelconverter"
    executable.write_text(
        "#!/bin/bash\n"
        "trap 'printf TERM > \"$SIGNAL_FILE\"; exit 42' TERM\n"
        'printf ready > "$READY_FILE"\n'
        "while true; do sleep 0.05; done\n"
    )
    executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "READY_FILE": str(ready_file),
        "SIGNAL_FILE": str(signal_file),
    }

    process = subprocess.Popen(
        ["docker/rvc2/entrypoint.sh", "convert", "rvc2"],
        env=env,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_file.exists()

        process.terminate()
        assert process.wait(timeout=5) == 42
        assert signal_file.read_text() == "TERM"
    finally:
        if process.poll() is None:
            process.kill()
