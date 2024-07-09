import shutil
import subprocess
import time
from logging import getLogger
from typing import Any, List, Union

from .exceptions import SubprocessException


def subprocess_run(
    cmd: Union[str, List[Any]],
    *,
    silent=False,
) -> subprocess.CompletedProcess:
    """Wrapper around `subprocess.run` that logs the command and its output.

    @type cmd: Union[str, List[Any]]
    @param cmd: Command to execute. Can be a string or a list of arguments.
    @type silent: bool
    @param silent: If True, the command will not be logged.
    """
    if isinstance(cmd, str):
        args = cmd.split()
    else:
        args = [str(arg) for arg in cmd]
        cmd = " ".join(args)
    logger = getLogger(__name__)
    cmd_name = args[0]

    if shutil.which(cmd_name) is None:
        raise SubprocessException(
            f"Command `{cmd_name}` not found. "
            "Ensure it is installed and in your PATH."
        )

    if not silent:
        logger.info(f"Executing `{cmd}`")
    start_time = time.time()

    result = subprocess.run(
        args, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    t = time.time() - start_time

    info_string = (
        f"Command `{cmd_name}` finished in {t:.2f} seconds "
        f"with return code {result.returncode}."
    )
    if result.returncode != 0:
        log_message = logger.error
    else:
        log_message = logger.info

    if not silent:
        log_message(info_string)
    if result.stderr:
        string = result.stderr.decode(errors="ignore")
        if not silent:
            log_message(f"[ STDERR ]:\n{string}")
        info_string += f"\n[ STDERR ]:\n{string}"
    if result.stdout:
        string = result.stdout.decode(errors="ignore")
        if not silent:
            log_message(f"[ STDOUT ]:\n{string}")
        info_string += f"\n[ STDOUT ]:\n{string}"
    if result.returncode != 0:
        raise SubprocessException(info_string)

    return result
