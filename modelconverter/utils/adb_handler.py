import subprocess
from typing import Optional, Tuple


class AdbHandler:
    def __init__(self, device_id: Optional[str] = None) -> None:
        self.device_args = ["-s", device_id] if device_id else []

    def _adb_run(self, args, **kwargs) -> Tuple[int, str, str]:
        subprocess.run(
            ["adb", "root"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        result = subprocess.run(
            ["adb", *self.device_args, *args],
            **kwargs,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout = result.stdout
        stderr = result.stderr
        assert result.returncode is not None
        if result.returncode != 0:
            raise RuntimeError(
                f"adb command {args[0]} failed with code {result.returncode}:\n"
                f"args: {args}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}\n"
            )
        return (
            result.returncode,
            stdout.decode(errors="ignore"),
            stderr.decode(errors="ignore"),
        )

    def shell(self, cmd: str) -> Tuple[int, str, str]:
        return self._adb_run(
            ["shell", cmd],
        )

    def pull(self, src: str, dst: str) -> Tuple[int, str, str]:
        return self._adb_run(["pull", src, dst])

    def push(self, src: str, dst: str) -> Tuple[int, str, str]:
        return self._adb_run(["push", src, dst])
