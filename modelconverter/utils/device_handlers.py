import re
import subprocess
from abc import ABC, abstractmethod

from loguru import logger
from luxonis_ml.typing import PathType
from typing_extensions import override


class DeviceHandler(ABC):
    """Abstract interface for communicating with a device.

    Implementations provide shell access and file transfer operations
    over a concrete transport such as SSH or ADB.
    """

    def __init__(self, silent: bool = True) -> None:
        """Initialize the device handler.

        Args:
            silent: If ``True``, suppress command logging by default.

        """
        self.silent = silent

    @abstractmethod
    def shell(
        self, cmd: str, *, check: bool = True, silent: bool | None = None
    ) -> tuple[int, str, str]:
        """Execute a shell command on the target device.

        Args:
            cmd: Shell command to execute on the device.
            check: If ``True``, propagate subprocess failures as
                exceptions.
            silent: If ``True``, suppress command logging. If ``None``,
                use the default logging behavior of the handler.

        Returns:
            A tuple containing return code, stdout, and stderr.

        """

    @abstractmethod
    def pull(
        self, src: PathType, dst: PathType, *, check: bool = True
    ) -> tuple[int, str, str]:
        """Copy a file or directory from the device to the local
        machine.

        Args:
            src: Source path on the device.
            dst: Destination path on the local machine.
            check: If ``True``, propagate subprocess failures as
                exceptions.

        Returns:
            A tuple containing return code, stdout, and stderr.

        """

    @abstractmethod
    def push(
        self, src: PathType, dst: PathType, *, check: bool = True
    ) -> tuple[int, str, str]:
        """Copy a file or directory from the local machine to the
        device.

        Args:
            src: Source path on the local machine.
            dst: Destination path on the device.
            check: If ``True``, propagate subprocess failures as
                exceptions.

        Returns:
            A tuple containing return code, stdout, and stderr.

        """

    def run(
        self, *args, check: bool = False, silent: bool = True, **kwargs
    ) -> tuple[int, str, str]:
        """Run a subprocess command and return its result.

        The command arguments are converted to strings before execution.
        Output is always captured and decoded using a best-effort
        strategy.

        Args:
            *args: Positional command arguments passed to
                ``subprocess.run``.
            check: If ``True``, propagate subprocess failures as
                exceptions.
            silent: If ``True``, suppress command logging.
            **kwargs: Additional keyword arguments forwarded to
                ``subprocess.run``.

        Returns:
            A tuple containing return code, stdout, and stderr.

        Raises:
            subprocess.CalledProcessError: If ``check`` is enabled and
                the subprocess exits with a non-zero status.

        """
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
    """Device handler implementation based on SSH and SCP."""

    def __init__(self, ip: str, silent: bool = True) -> None:
        """Initialize an SSH handler for the target device.

        Args:
            ip: Target device IP address.
            silent: If ``True``, suppress command logging.

        """
        super().__init__(silent)
        self._address = f"root@{ip}"

    @override
    def shell(
        self, cmd: str, *, check: bool = True, silent: bool | None = None
    ) -> tuple[int, str, str]:
        return self.run(
            "ssh",
            self._address,
            cmd,
            check=check,
            silent=silent if silent is not None else self.silent,
        )

    @override
    def pull(
        self, src: PathType, dst: PathType, *, check: bool = True
    ) -> tuple[int, str, str]:
        return self.run(
            "scp",
            "-r",
            f"{self._address}:{src}",
            dst,
            check=check,
            silent=self.silent,
        )

    @override
    def push(
        self, src: PathType, dst: PathType, *, check: bool = True
    ) -> tuple[int, str, str]:
        return self.run(
            "scp",
            "-r",
            src,
            f"{self._address}:{dst}",
            check=check,
            silent=self.silent,
        )


class AdbHandler(DeviceHandler):
    """Device handler implementation based on Android Debug Bridge."""

    def __init__(
        self, device_id: str | None = None, silent: bool = True
    ) -> None:
        """Initialize an ADB handler for the target device.

        If no device ID is provided, the first connected device is
        selected.

        Args:
            device_id: Optional ADB device identifier.
            silent: If ``True``, suppress command logging.

        Raises:
            RuntimeError: If device enumeration fails or no connected
                device is available.
            ValueError: If the specified device is not connected.

        """
        super().__init__(silent)
        device_id = self._check_adb_connection(device_id)
        self._device_args = ["-s", device_id] if device_id else []

    @override
    def run(self, *args, check: bool = True, **kwargs) -> tuple[int, str, str]:
        subprocess.run(
            ["adb", *map(str, self._device_args), "root"],
            capture_output=True,
            check=check,
        )
        return super().run(
            "adb", *self._device_args, *args, check=check, **kwargs
        )

    @override
    def shell(
        self, cmd: str, *, check: bool = True, silent: bool | None = None
    ) -> tuple[int, str, str]:
        return self.run(
            "shell",
            cmd,
            check=check,
            silent=silent if silent is not None else self.silent,
        )

    @override
    def pull(
        self, src: PathType, dst: PathType, *, check: bool = True
    ) -> tuple[int, str, str]:
        return self.run("pull", src, dst, check=check)

    @override
    def push(
        self, src: PathType, dst: PathType, *, check: bool = True
    ) -> tuple[int, str, str]:
        return self.run("push", src, dst, check=check)

    def _check_adb_connection(self, device_id: str | None) -> str:
        """Validate ADB connectivity and resolve the device ID.

        Args:
            device_id: Requested device identifier, or ``None`` to
                auto-select the first connected device.

        Returns:
            Resolved connected device identifier.

        Raises:
            RuntimeError: If device enumeration fails or no device is
                connected.
            ValueError: If the requested device is not connected.

        """
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


def create_handler(
    device_ip: str | None, device_adb_id: str | None
) -> DeviceHandler:
    try:
        handler = AdbHandler(device_adb_id)
    except Exception:
        if device_ip is not None:
            handler = SSHHandler(device_ip)
        else:
            raise
    return handler
