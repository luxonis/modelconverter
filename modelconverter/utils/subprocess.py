import io
import shutil
import subprocess
import threading
import time

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


def subprocess_run(
    cmd: str | list[str], *, silent: bool = False
) -> SubprocessResult:
    """Wrapper around subprocess.run that logs, raises on error, and
    tracks peak RAM usage without hanging on stdout/stderr.

    @param cmd: Command to execute. String or list of arguments.
    @param silent: If True, suppress logs.
    @return: SubprocessResult with .peak_memory attribute.
    """
    if isinstance(cmd, str):
        args = cmd.split()
    else:
        args = [str(arg) for arg in cmd]
        cmd = " ".join(args)
    cmd_name = args[0]

    if shutil.which(cmd_name) is None:
        raise SubprocessException(
            f"Command `{cmd_name}` not found. Ensure it is installed and in your PATH."
        )

    if not silent:
        logger.info(f"Executing `{cmd}`")
    start_time = time.time()

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    ps_proc = psutil.Process(proc.pid)
    peak_mem = 0

    stdout_buf, stderr_buf = [], []

    def _reader(stream: io.TextIOWrapper, buf: list[str]) -> None:
        for line in iter(stream.readline, ""):
            buf.append(line)
            if not silent:
                logger.info(line.strip())
        stream.close()

    threads = [
        threading.Thread(target=_reader, args=(proc.stdout, stdout_buf)),
        threading.Thread(target=_reader, args=(proc.stderr, stderr_buf)),
    ]
    for t in threads:
        t.daemon = True
        t.start()

    while proc.poll() is None:
        try:
            mem = ps_proc.memory_info().rss
            peak_mem = max(peak_mem, mem)
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)

    for t in threads:
        t.join(timeout=1.0)

    t = time.time() - start_time
    result = SubprocessResult(
        args,
        proc.returncode,
        "".join(stdout_buf).encode(),
        "".join(stderr_buf).encode(),
        peak_memory=peak_mem,
        total_time=t,
    )

    info_string = (
        f"Command `{cmd_name}` finished in {t:.2f} seconds "
        f"with return code {result.returncode}."
    )
    log_message = logger.error if result.returncode != 0 else logger.info

    if not silent:
        log_message(info_string)

    if result.returncode != 0:
        raise SubprocessException(info_string)

    return result
