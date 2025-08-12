import re
import subprocess

from loguru import logger
from luxonis_ml.typing import PathType


class AdbHandler:
    def __init__(
        self, device_id: str | None = None, silent: bool = True
    ) -> None:
        device_id = self._check_adb_connection(device_id)
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

    def _check_adb_connection(self, device_id: str | None) -> str:
        result = subprocess.run(
            ["adb", "devices"], check=False, capture_output=True
        )
        if result.returncode == 0:
            pattern = re.compile(r"^(\w+)\s+device$", re.MULTILINE)
            devices = pattern.findall(result.stdout.decode())
        else:
            raise RuntimeError("Unable to verify device ID")

        if device_id is None:
            if len(devices) == 0:
                raise RuntimeError("No devices connected")
            logger.warning(
                "No device ID specified, using the first connected "
                f"device: {devices[0]}"
            )
            return devices[0]
        if device_id not in devices:
            raise ValueError(
                f"Device ID '{device_id}' not found in connected devices: {devices}"
            )
        logger.info(f"Using device ID: {device_id}")

        return device_id
