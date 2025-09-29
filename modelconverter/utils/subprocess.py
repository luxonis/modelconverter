import io
import shutil
import subprocess
import threading
import time
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


class SubprocessHandle:
    """Context manager wrapping a subprocess with live psutil access and
    deferred result collection."""

    def __init__(
        self,
        args: list[str],
        *,
        silent: bool = False,
        timeout: float | None = None,
    ):
        self.args = args
        self.cmd_name = args[0]
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
            logger.info(f"Executing `{' '.join(self.args)}`")

        self._start_time = time.time()
        self._proc = subprocess.Popen(
            self.args,
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
            self.wait()
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
            self.args,
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
    cmd: str | list[Any], *, silent: bool = False
) -> SubprocessResult:
    """Backwards-compatible wrapper.

    Blocks until done and returns result.
    """
    if isinstance(cmd, str):
        args = cmd.split()
    else:
        args = [str(arg) for arg in cmd]

    with SubprocessHandle(args, silent=silent) as handle:
        while handle.poll() is None:
            time.sleep(0.1)
        return handle.result()
