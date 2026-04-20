import re
import subprocess
from abc import ABC, abstractmethod

from loguru import logger
from luxonis_ml.typing import PathType
from typing_extensions import override


class DeviceHandler(ABC):
    @abstractmethod
    def shell(self, cmd: str, *, check: bool = False) -> tuple[int, str, str]:
        pass

    @abstractmethod
    def pull(
        self, src: PathType, dst: PathType, *, check: bool = False
    ) -> tuple[int, str, str]:
        pass

    @abstractmethod
    def push(
        self, src: PathType, dst: PathType, *, check: bool = False
    ) -> tuple[int, str, str]:
        pass

    def run(
        self, *args, check: bool = False, silent: bool = False, **kwargs
    ) -> tuple[int, str, str]:
        args = list(map(str, args))
        if not silent:
            logger.info(f"{' '.join(args)}")
        result = subprocess.run(
            args,
            **kwargs,
            capture_output=True,
            check=check,
        )
        stdout = result.stdout
        stderr = result.stderr
        assert result.returncode is not None
        return (
            result.returncode,
            stdout.decode(errors="ignore"),
            stderr.decode(errors="ignore"),
        )


class SSHHandler(DeviceHandler):
    def __init__(self, ip: str, silent: bool = True) -> None:
        self._address = f"root@{ip}"
        self.silent = silent

    @override
    def shell(self, cmd: str, *, check: bool = False) -> tuple[int, str, str]:
        return self.run(
            "ssh",
            self._address,
            cmd,
            check=check,
            silent=self.silent,
        )

    @override
    def pull(
        self, src: PathType, dst: PathType, *, check: bool = False
    ) -> tuple[int, str, str]:
        return self.run(
            "scp",
            f"{self._address}:{src}",
            dst,
            check=check,
            silent=self.silent,
        )

    @override
    def push(
        self, src: PathType, dst: PathType, *, check: bool = False
    ) -> tuple[int, str, str]:
        return self.run(
            "scp",
            src,
            f"{self._address}:{dst}",
            check=check,
            silent=self.silent,
        )


class AdbHandler(DeviceHandler):
    def __init__(
        self, device_id: str | None = None, silent: bool = True
    ) -> None:
        device_id = self._check_adb_connection(device_id)
        self._device_args = ["-s", device_id] if device_id else []
        self.silent = silent

    @override
    def run(self, *args, check: bool, **kwargs) -> tuple[int, str, str]:
        subprocess.run(
            ["adb", *map(str, self._device_args), "root"],
            capture_output=True,
            check=check,
        )
        return super().run(
            "adb", *self._device_args, *args, check=check, **kwargs
        )

    @override
    def shell(self, cmd: str, *, check: bool = False) -> tuple[int, str, str]:
        return self.run("shell", cmd, check=check)

    @override
    def pull(
        self, src: PathType, dst: PathType, *, check: bool = False
    ) -> tuple[int, str, str]:
        return self.run("pull", src, dst, check=check)

    @override
    def push(
        self, src: PathType, dst: PathType, check: bool = False
    ) -> tuple[int, str, str]:
        return self.run("push", src, dst, check=check)

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
                f"device: {devices[0]}."
            )
            return devices[0]
        if device_id not in devices:
            raise ValueError(
                f"Device ID '{device_id}' not found in connected devices: {devices}. Please check the device connection."
            )
        logger.info(f"Using device ID: {device_id}")

        return device_id
