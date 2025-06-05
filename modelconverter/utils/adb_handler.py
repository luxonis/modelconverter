import subprocess

from loguru import logger
from luxonis_ml.typing import PathType


class AdbHandler:
    def __init__(
        self, device_id: str | None = None, silent: bool = True
    ) -> None:
        self.device_args = ["-s", device_id] if device_id else []
        self.silent = silent

    def _adb_run(self, *args, **kwargs) -> tuple[int, str, str]:
        subprocess.run(
            ["adb", *map(str, self.device_args), "root"],
            capture_output=True,
            check=False,
        )
        if not self.silent:
            logger.info(f"Executing adb command: {' '.join(map(str, args))}")
        result = subprocess.run(
            ["adb", *self.device_args, *args],
            **kwargs,
            capture_output=True,
            check=False,
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

    def shell(self, cmd: str) -> tuple[int, str, str]:
        return self._adb_run("shell", cmd)

    def pull(self, src: PathType, dst: PathType) -> tuple[int, str, str]:
        return self._adb_run("pull", src, dst)

    def push(self, src: PathType, dst: PathType) -> tuple[int, str, str]:
        return self._adb_run("push", src, dst)
