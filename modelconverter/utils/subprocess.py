"""Execution of external commands used by the conversions.

The conversions drive vendor toolchains through their command line
tools. The helpers here run such a command, stream its output to the
log with ANSI escape sequences removed, and record the peak memory
usage and the wall-clock run time of the whole process tree.
"""

import io
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import suppress
from types import TracebackType

import psutil
from loguru import logger
from luxonis_ml.typing import PathType
from typing_extensions import Self


class SubprocessResult(subprocess.CompletedProcess[bytes]):
    """Extension of ``subprocess.CompletedProcess`` that also carries peak
    memory usage.

    Attributes:
        peak_memory: Peak memory usage of the process and its children,
            in bytes.
        total_time: Wall-clock run time of the process, in seconds.

    """

    def __init__(self, *args, peak_memory: int, total_time: float, **kwargs):
        """Initialize the result.

        Args:
            *args: Positional arguments passed to
                ``subprocess.CompletedProcess``.
            peak_memory: Peak memory usage of the process and its
                children, in bytes.
            total_time: Wall-clock run time of the process, in seconds.
            **kwargs: Keyword arguments passed to
                ``subprocess.CompletedProcess``.

        """
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
        """Return the base representation with memory and run time."""
        base = super().__repr__()
        return (
            f"{base.rstrip(')')}, "
            f"peak_memory={self._human_memory()}, "
            f"total_time={round(self.total_time, 3)}s)"
        )

    def __str__(self) -> str:
        """Return the representation of the result."""
        return repr(self)

    def __rich_repr__(
        self,
    ) -> Iterator[tuple[str, list[str] | int | str | bytes]]:
        """Yield the fields used by ``rich`` to render the result.

        Yields:
            Tuples of field name and field value.

        """
        yield "args", self.args
        yield "returncode", self.returncode
        yield "stdout", self.stdout
        yield "stderr", self.stderr
        yield "peak_memory", self._human_memory()
        yield "total_time", f"{round(self.total_time, 3)}s"


class SubprocessHandle:
    """Context manager wrapping a subprocess with live psutil access and
    deferred result collection.
    """

    def __init__(
        self,
        cmd: str | Sequence[PathType],
        *,
        silent: bool = False,
        timeout: float | None = None,
    ):
        """Initialize the subprocess handle.

        Args:
            cmd: Command to execute. If a string is given, it will be
                split on whitespace. If a list is given, each element
                will be converted to a string.
            silent: If ``True``, suppress all output from the command.
            timeout: If given, the maximum time in seconds to allow the
                process to run. If the timeout is exceeded, the process
                is terminated and ``subprocess.TimeoutExpired`` is raised.

        """
        if isinstance(cmd, str):
            self._cmd = cmd.split()
        else:
            self._cmd = [str(arg) for arg in cmd]

        self._cmd_name = self._cmd[0]
        self._silent = silent
        self._peak_mem: int = 0
        self._stdout_buf: list[str] = []
        self._stderr_buf: list[str] = []
        self._timeout = timeout

        self._threads: list[threading.Thread] = []
        self._start_time: float = 0.0
        self._proc: subprocess.Popen | None = None
        self._ps_proc: psutil.Process | None = None

    @property
    def proc(self) -> subprocess.Popen:
        """Return the underlying ``subprocess.Popen`` object.

        Raises:
            RuntimeError: If the process has not been started yet.

        """
        if self._proc is None:
            raise RuntimeError(
                "Process not started yet. "
                "You must use `SubprocessHandle` as a context manager."
            )
        return self._proc

    @property
    def ps_proc(self) -> psutil.Process:
        """Return the ``psutil.Process`` wrapping the process.

        Raises:
            RuntimeError: If the process has not been started yet.

        """
        if self._ps_proc is None:
            raise RuntimeError(
                "Process not started yet. "
                "You must use `SubprocessHandle` as a context manager."
            )
        return self._ps_proc

    def __bool__(self) -> bool:
        """Return whether the process is still running.

        .. warning::
            Truth-testing the handle is not side-effect free: a
            ``while handle:`` loop is what enforces the timeout.

        Raises:
            subprocess.TimeoutExpired: If the configured timeout has
                been exceeded. The process is terminated first.

        """
        if time.time() - self._start_time > (self._timeout or float("inf")):
            with suppress(psutil.NoSuchProcess):
                self.ps_proc.terminate()
            raise subprocess.TimeoutExpired(
                self._cmd,
                self._timeout or 0,
                output="".join(self._stdout_buf).encode(),
                stderr="".join(self._stderr_buf).encode(),
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
        """Check whether the process has terminated.

        Returns:
            The return code, or ``None`` if the process is still
            running.

        """
        return self.proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the process to terminate.

        Args:
            timeout: Maximum time in seconds to wait. Ignored if a
                timeout was given when the handle was created.

        Returns:
            The return code of the process.

        """
        return self.proc.wait(timeout=self._timeout or timeout)

    def __enter__(self) -> Self:
        """Start the process and begin collecting its output.

        Spawns one reader thread per output stream and a thread that
        keeps track of the peak memory usage of the process tree.

        Returns:
            The handle itself.

        Raises:
            subprocess.SubprocessError: If the command is not found on
                the ``PATH``.

        """
        if shutil.which(self._cmd_name) is None:
            raise subprocess.SubprocessError(
                f"Command `{self._cmd_name}` not found. Ensure it is in PATH."
            )

        if not self._silent:
            logger.info(f"Executing `{' '.join(self._cmd)}`")

        self._start_time = time.time()
        self._proc = subprocess.Popen(
            self._cmd,
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
                if not self._silent:
                    logger.info(line.strip())
            stream.close()

        def _memory_monitor() -> None:
            while self.poll() is None:
                self._monitor_memory(interval=0.1)

        self._threads = [
            threading.Thread(
                target=_reader,
                args=(self._proc.stdout, self._stdout_buf),
                daemon=True,
            ),
            threading.Thread(
                target=_reader,
                args=(self._proc.stderr, self._stderr_buf),
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
        """Wait for the process and its reader threads to finish.

        Args:
            exc_type: Type of the exception raised in the block, if any.
            exc_value: Exception raised in the block, if any.
            traceback: Traceback of the exception raised in the block,
                if any.

        """
        if self.poll() is None:
            self.wait(self._timeout)
        for t in self._threads:
            t.join(timeout=1.0)

    def result(self) -> SubprocessResult:
        """Collect the result of the finished process.

        Waits for the reader threads to finish and logs a summary of
        the run unless the handle is silent.

        Returns:
            Result of the command, carrying its captured output, the
            peak memory usage and the total run time.

        Raises:
            subprocess.SubprocessError: If the command finished with a
                non-zero return code.

        """
        for t in self._threads:
            t.join(timeout=1.0)
        total_time = time.time() - self._start_time
        res = SubprocessResult(
            self._cmd,
            self.proc.returncode,
            "".join(self._stdout_buf).encode(),
            "".join(self._stderr_buf).encode(),
            peak_memory=self._peak_mem,
            total_time=total_time,
        )
        info_string = (
            f"Command `{self._cmd_name}` finished in {total_time:.2f} s "
            f"with return code {res.returncode}."
        )
        log_message = logger.error if res.returncode != 0 else logger.info
        if not self._silent:
            log_message(info_string)
        if res.returncode != 0:
            raise subprocess.SubprocessError(info_string)
        return res

    def _current_memory(self) -> int:
        """Return current memory usage of the process and its children."""
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

    def _monitor_memory(self, interval: float = 0.1) -> None:
        """Call periodically to update peak memory usage."""
        try:
            self._peak_mem = max(self._peak_mem, self._current_memory())
            time.sleep(interval)
        except psutil.NoSuchProcess:
            pass


def subprocess_run(
    cmd: str | Sequence[PathType],
    *,
    silent: bool = False,
    timeout: float | None = None,
) -> SubprocessResult:
    """Run a command and block until it finishes.

    Backwards-compatible wrapper around `SubprocessHandle`.

    Args:
        cmd: Command to execute. If a string is given, it will be split
            on whitespace. If a list is given, each element will be
            converted to a string.
        silent: If ``True``, suppress all output from the command.
        timeout: If given, the maximum time in seconds to allow the
            process to run. If the timeout is exceeded, the process is
            terminated and ``subprocess.TimeoutExpired`` is raised.

    Returns:
        Result of the command.

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
    r"""Remove ANSI escape sequences from a string.

    Args:
        s: String to strip.

    Returns:
        The string without ANSI escape sequences.

    Example:
        >>> strip_ansi("\x1b[31mred\x1b[0m")
        'red'

    """
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", s)
