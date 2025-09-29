import io
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from types import TracebackType
from typing import Any

import psutil
from loguru import logger

from .exceptions import SubprocessException


class SubprocessResult(subprocess.CompletedProcess):
    """Extension of subprocess.CompletedProcess that also carries peak
    memory usage."""

    def __init__(self, *args, peak_memory: int, total_time: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_memory = peak_memory
        self.total_time = total_time

    def _human_memory(self) -> str:
        """Return human-readable peak memory usage."""
        units = ["B", "KB", "MB"]
        mem = self.peak_memory
        for unit in units:
            if mem < 1024:
                return f"{mem:.2f} {unit}"
            mem /= 1024
        return f"{mem:.2f} GB"

    def __repr__(self) -> str:
        base = super().__repr__()
        return (
            f"{base.rstrip(')')}, "
            f"peak_memory={self._human_memory()}, "
            f"total_time={round(self.total_time, 3)}s)"
        )

    def __str__(self) -> str:
        return repr(self)

    def __rich_repr__(self) -> Iterator[tuple[str, Any]]:
        yield "args", self.args
        yield "returncode", self.returncode
        yield "stdout", self.stdout
        yield "stderr", self.stderr
        yield "peak_memory", self._human_memory()
        yield "total_time", f"{round(self.total_time, 3)}s"


class SubprocessHandle:
    """Context manager wrapping a subprocess with live psutil access and
    deferred result collection."""

    def __init__(
        self,
        cmd: str | list[Any],
        *,
        silent: bool = False,
        timeout: float | None = None,
    ):
        """Initialize the subprocess handle.

        @type args: str | list[Any]
        @param args: Command to execute. If a string is given, it will
            be split on whitespace. If a list is given, each element
            will be converted to a string.
        @type silent: bool
        @param silent: If True, suppress all output from the command.
        @type timeout: float | None
        @param timeout: If given, the maximum time in seconds to allow
            the process to run. If the timeout is exceeded, the process
            will be
        """

        if isinstance(cmd, str):
            self.cmd = cmd.split()
        else:
            self.cmd = [str(arg) for arg in cmd]

        self.cmd_name = self.cmd[0]
        self.silent = silent
        self.peak_mem: int = 0
        self.stdout_buf: list[str] = []
        self.stderr_buf: list[str] = []
        self.timeout = timeout

        self._threads: list[threading.Thread] = []
        self._start_time: float = 0.0
        self._proc: subprocess.Popen | None = None
        self._ps_proc: psutil.Process | None = None

    @property
    def proc(self) -> subprocess.Popen:
        if self._proc is None:
            raise RuntimeError(
                "Process not started yet. "
                "You must use `SubprocessHandle` as a context manager."
            )
        return self._proc

    @property
    def ps_proc(self) -> psutil.Process:
        if self._ps_proc is None:
            raise RuntimeError(
                "Process not started yet. "
                "You must use `SubprocessHandle` as a context manager."
            )
        return self._ps_proc

    def __bool__(self) -> bool:
        """Return whether the process is still running.

        Also checks for timeout and raises TimeoutExpired if exceeded.
        """
        if time.time() - self._start_time > (self.timeout or float("inf")):
            with suppress(psutil.NoSuchProcess):
                self.ps_proc.terminate()
            raise subprocess.TimeoutExpired(
                self.cmd,
                self.timeout or 0,
                output="".join(self.stdout_buf).encode(),
                stderr="".join(self.stderr_buf).encode(),
            )
        return self.poll() is None

    def is_suspended(self) -> bool:
        """Return whether the process is currently suspended."""
        try:
            return self.ps_proc.status() == psutil.STATUS_STOPPED
        except psutil.NoSuchProcess:
            return False

    def suspend(self) -> None:
        """Suspend the process."""
        with suppress(psutil.NoSuchProcess):
            self.ps_proc.suspend()

    def resume(self) -> None:
        """Resume the process."""
        with suppress(psutil.NoSuchProcess):
            self.ps_proc.resume()

    def poll(self) -> int | None:
        """Check if the process has terminated.

        Returns returncode or None.
        """
        return self.proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process to terminate.

        Returns returncode.
        """
        return self.proc.wait(timeout=self.timeout or timeout)

    def __enter__(self) -> "SubprocessHandle":
        if shutil.which(self.cmd_name) is None:
            raise SubprocessException(
                f"Command `{self.cmd_name}` not found. Ensure it is in PATH."
            )

        if not self.silent:
            logger.info(f"Executing `{' '.join(self.cmd)}`")

        self._start_time = time.time()
        self._proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        self._ps_proc = psutil.Process(self._proc.pid)

        def _reader(stream: io.TextIOWrapper, buf: list[str]) -> None:
            for line in iter(stream.readline, ""):
                line = strip_ansi(line)
                buf.append(line)
                if not self.silent:
                    logger.info(line.strip())
            stream.close()

        def _memory_monitor() -> None:
            while self.poll() is None:
                self.monitor_memory(interval=0.1)

        self._threads = [
            threading.Thread(
                target=_reader,
                args=(self._proc.stdout, self.stdout_buf),
                daemon=True,
            ),
            threading.Thread(
                target=_reader,
                args=(self._proc.stderr, self.stderr_buf),
                daemon=True,
            ),
            threading.Thread(target=_memory_monitor, daemon=True),
        ]
        for t in self._threads:
            t.start()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.poll() is None:
            self.wait(self.timeout)
        for t in self._threads:
            t.join(timeout=1.0)

    def current_memory(self) -> int:
        """Return current memory usage of the process and its
        children."""
        if self._ps_proc is None:
            return 0
        try:
            mem = self._ps_proc.memory_info().rss
            for child in self._ps_proc.children(recursive=True):
                with suppress(psutil.NoSuchProcess):
                    mem += child.memory_info().rss
        except psutil.NoSuchProcess:
            return 0
        else:
            return mem

    def monitor_memory(self, interval: float = 0.1) -> None:
        """Call periodically to update peak memory usage."""
        try:
            self.peak_mem = max(self.peak_mem, self.current_memory())
            time.sleep(interval)
        except psutil.NoSuchProcess:
            pass

    def result(self) -> SubprocessResult:
        for t in self._threads:
            t.join(timeout=1.0)
        total_time = time.time() - self._start_time
        res = SubprocessResult(
            self.cmd,
            self.proc.returncode,
            "".join(self.stdout_buf).encode(),
            "".join(self.stderr_buf).encode(),
            peak_memory=self.peak_mem,
            total_time=total_time,
        )
        info_string = (
            f"Command `{self.cmd_name}` finished in {total_time:.2f} s "
            f"with return code {res.returncode}."
        )
        log_message = logger.error if res.returncode != 0 else logger.info
        if not self.silent:
            log_message(info_string)
        if res.returncode != 0:
            raise SubprocessException(info_string)
        return res


def subprocess_run(
    cmd: str | list[Any], *, silent: bool = False, timeout: float | None = None
) -> SubprocessResult:
    """Backwards-compatible wrapper.

    Blocks until done and returns result.
    @type cmd: str | list[Any]
    @param cmd: Command to execute. If a string is given, it will be
        split on whitespace. If a list is given, each element will be
        converted to a string.
    @type silent: bool
    @param silent: If True, suppress all output from the command.
    @type timeout: float | None
    @param timeout: If given, the maximum time in seconds to allow the
        process to run. If the timeout is exceeded, the process will be
        terminated and a TimeoutExpired exception will be raised.
    @rtype: SubprocessResult
    @return: Result of the command.
    """
    if isinstance(cmd, str):
        args = cmd.split()
    else:
        args = [str(arg) for arg in cmd]

    with SubprocessHandle(args, silent=silent, timeout=timeout) as proc:
        while proc:
            time.sleep(0.1)
        return proc.result()


def strip_ansi(s: str) -> str:
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", s)
