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

    @abstractmethod
    def shell(self, cmd: str, *, check: bool = False) -> tuple[int, str, str]:
        """Execute a shell command on the target device.

        @param cmd: Shell command to execute on the device.
        @type cmd: str
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str]
        """

    @abstractmethod
    def pull(
        self, src: PathType, dst: PathType, *, check: bool = False
    ) -> tuple[int, str, str]:
        """Copy a file or directory from the device to the local
        machine.

        @param src: Source path on the device.
        @type src: PathType
        @param dst: Destination path on the local machine.
        @type dst: PathType
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str]
        """

    @abstractmethod
    def push(
        self, src: PathType, dst: PathType, *, check: bool = False
    ) -> tuple[int, str, str]:
        """Copy a file or directory from the local machine to the
        device.

        @param src: Source path on the local machine.
        @type src: PathType
        @param dst: Destination path on the device.
        @type dst: PathType
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str]
        """

    def run(
        self, *args, check: bool = False, silent: bool = False, **kwargs
    ) -> tuple[int, str, str]:
        """Run a subprocess command and return its result.

        The command arguments are converted to strings before execution.
        Output is always captured and decoded using a best-effort
        strategy.

        @param args: Positional command arguments passed to
            C{subprocess.run()}.
        @type args: tuple
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @param silent: If C{True}, suppress command logging.
        @type silent: bool
        @param kwargs: Additional keyword arguments forwarded to
            C{subprocess.run()}.
        @type kwargs: dict
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str] @raises
            subprocess.CalledProcessError: If C{check} is enabled and
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
        """Initialize the SSH handler.

        @param ip: Target device IP address.
        @type ip: str
        @param silent: If C{True}, suppress command logging.
        @type silent: bool
        """
        self._address = f"root@{ip}"
        self.silent = silent

    @override
    def shell(self, cmd: str, *, check: bool = False) -> tuple[int, str, str]:
        """Execute a shell command on the remote device over SSH.

        @param cmd: Shell command to execute remotely.
        @type cmd: str
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str] @raises
            subprocess.CalledProcessError: If C{check} is enabled and
            the SSH command exits with a non-zero status.
        """
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
        """Copy a file or directory from the remote device using SCP.

        @param src: Source path on the remote device.
        @type src: PathType
        @param dst: Destination path on the local machine.
        @type dst: PathType
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str] @raises
            subprocess.CalledProcessError: If C{check} is enabled and
            the SCP command exits with a non-zero status.
        """
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
        """Copy a file or directory to the remote device using SCP.

        @param src: Source path on the local machine.
        @type src: PathType
        @param dst: Destination path on the remote device.
        @type dst: PathType
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str] @raises
            subprocess.CalledProcessError: If C{check} is enabled and
            the SCP command exits with a non-zero status.
        """
        return self.run(
            "scp",
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
        """Initialize the ADB handler.

        If no device ID is provided, the first connected device is
        selected.

        @param device_id: Optional ADB device identifier.
        @type device_id: str | None
        @param silent: If C{True}, suppress command logging.
        @type silent: bool
        @raises RuntimeError: If device enumeration fails or no
            connected device is available.
        @raises ValueError: If the specified device is not connected.
        """
        device_id = self._check_adb_connection(device_id)
        self._device_args = ["-s", device_id] if device_id else []
        self.silent = silent

    @override
    def run(self, *args, check: bool, **kwargs) -> tuple[int, str, str]:
        """Run an ADB command after requesting root access.

        The method first invokes C{adb root} for the selected device and
        then executes the requested ADB subcommand.

        @param args: ADB subcommand arguments.
        @type args: tuple
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @param kwargs: Additional keyword arguments forwarded to the
            base implementation.
        @type kwargs: dict
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str] @raises
            subprocess.CalledProcessError: If C{check} is enabled and
            either C{adb root} or the requested ADB command fails.
        """
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
        """Execute a shell command on the ADB-connected device.

        @param cmd: Shell command to execute on the device.
        @type cmd: str
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str] @raises
            subprocess.CalledProcessError: If C{check} is enabled and
            the ADB shell command exits with a non-zero status.
        """
        return self.run("shell", cmd, check=check)

    @override
    def pull(
        self, src: PathType, dst: PathType, *, check: bool = False
    ) -> tuple[int, str, str]:
        """Copy a file or directory from the device using ADB.

        @param src: Source path on the device.
        @type src: PathType
        @param dst: Destination path on the local machine.
        @type dst: PathType
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str] @raises
            subprocess.CalledProcessError: If C{check} is enabled and
            the ADB pull command exits with a non-zero status.
        """
        return self.run("pull", src, dst, check=check)

    @override
    def push(
        self, src: PathType, dst: PathType, check: bool = False
    ) -> tuple[int, str, str]:
        """Copy a file or directory to the device using ADB.

        @param src: Source path on the local machine.
        @type src: PathType
        @param dst: Destination path on the device.
        @type dst: PathType
        @param check: If C{True}, propagate subprocess failures as
            exceptions.
        @type check: bool
        @return: A tuple containing return code, stdout, and stderr.
        @rtype: tuple[int, str, str] @raises
            subprocess.CalledProcessError: If C{check} is enabled and
            the ADB push command exits with a non-zero status.
        """
        return self.run("push", src, dst, check=check)

    def _check_adb_connection(self, device_id: str | None) -> str:
        """Validate ADB connectivity and resolve the device ID.

        @param device_id: Requested device identifier, or C{None} to
            auto-select the first connected device.
        @type device_id: str | None
        @return: Resolved connected device identifier.
        @rtype: str
        @raises RuntimeError: If device enumeration fails or no device
            is connected.
        @raises ValueError: If the requested device is not connected.
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
