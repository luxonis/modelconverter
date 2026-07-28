"""Unit tests for ``modelconverter.utils.subprocess``."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from modelconverter.utils.subprocess import (
    SubprocessHandle,
    SubprocessResult,
    strip_ansi,
    subprocess_run,
)


def _command(code: str) -> list[str]:
    return [sys.executable, "-c", code]


@pytest.mark.parametrize(
    ("peak_memory", "expected"),
    [
        (1, "1.00 B"),
        (1024, "1.00 KB"),
        (1024**2, "1.00 MB"),
        (1024**3, "1.00 GB"),
    ],
)
def test_result_presentation(peak_memory: int, expected: str):
    result = SubprocessResult(
        ["command"],
        0,
        b"stdout",
        b"stderr",
        peak_memory=peak_memory,
        total_time=1.2349,
    )

    assert result._human_memory() == expected
    assert expected in repr(result)
    assert str(result) == repr(result)
    assert dict(result.__rich_repr__())["peak_memory"] == expected


def test_handle_requires_context_manager():
    handle = SubprocessHandle(["command"])

    with pytest.raises(RuntimeError, match="Process not started"):
        _ = handle.proc
    with pytest.raises(RuntimeError, match="Process not started"):
        _ = handle.ps_proc
    assert handle.current_memory() == 0


def test_missing_command_is_reported():
    with (
        pytest.raises(subprocess.SubprocessError, match="not found"),
        SubprocessHandle(["definitely-not-a-command"]),
    ):
        pass


def test_subprocess_run_collects_cleaned_output():
    result = subprocess_run(
        _command(
            "import sys, time; "
            "print('\\x1b[31mstdout\\x1b[0m'); "
            "print('stderr', file=sys.stderr); "
            "time.sleep(0.15)"
        )
    )

    assert result.returncode == 0
    assert result.stdout == b"stdout\n"
    assert result.stderr == b"stderr\n"
    assert result.peak_memory > 0
    assert result.total_time > 0


def test_subprocess_run_accepts_string_command():
    result = subprocess_run(f"{sys.executable} -c print('ok')", silent=True)

    assert result.stdout == b"ok\n"


def test_handle_accepts_string_command():
    with SubprocessHandle(
        f"{sys.executable} -c print('ok')", silent=True
    ) as handle:
        handle.wait()

    assert handle.result().stdout == b"ok\n"


def test_failed_command_raises_subprocess_error():
    with pytest.raises(subprocess.SubprocessError, match="return code 2"):
        subprocess_run(_command("import sys; sys.exit(2)"), silent=True)


def test_timeout_terminates_process_and_preserves_output():
    def run() -> None:
        with SubprocessHandle(
            _command("import time; time.sleep(5)"), timeout=0.01
        ) as handle:
            handle._start_time = 0
            handle.stdout_buf.append("stdout")
            handle.stderr_buf.append("stderr")
            bool(handle)

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        run()

    assert exc_info.value.output == b"stdout"
    assert exc_info.value.stderr == b"stderr"


def test_process_control_and_memory_helpers(monkeypatch: pytest.MonkeyPatch):
    handle = SubprocessHandle(["command"])
    process = Mock()
    process.memory_info.return_value = SimpleNamespace(rss=10)
    child = Mock()
    child.memory_info.return_value = SimpleNamespace(rss=5)
    missing_child = Mock()
    missing_child.memory_info.side_effect = psutil.NoSuchProcess(1)
    process.children.return_value = [child, missing_child]
    handle._ps_proc = process

    assert handle.is_suspended() is False
    process.status.return_value = psutil.STATUS_STOPPED
    assert handle.is_suspended() is True
    handle.suspend()
    handle.resume()
    assert handle.current_memory() == 15

    process.memory_info.side_effect = psutil.NoSuchProcess(1)
    assert handle.current_memory() == 0
    process.memory_info.side_effect = None
    monkeypatch.setattr(handle, "current_memory", lambda: 20)
    handle.monitor_memory(interval=0)
    assert handle.peak_mem == 20
    monkeypatch.setattr(
        handle,
        "current_memory",
        Mock(side_effect=psutil.NoSuchProcess(1)),
    )
    handle.monitor_memory(interval=0)


def test_process_control_ignores_disappearing_process():
    handle = SubprocessHandle(["command"])
    process = Mock()
    process.status.side_effect = psutil.NoSuchProcess(1)
    process.suspend.side_effect = psutil.NoSuchProcess(1)
    process.resume.side_effect = psutil.NoSuchProcess(1)
    handle._ps_proc = process

    assert handle.is_suspended() is False
    handle.suspend()
    handle.resume()


def test_wait_uses_handle_timeout():
    handle = SubprocessHandle(["command"])
    process = Mock()
    process.wait.return_value = 0
    handle._proc = process

    assert handle.wait(timeout=2) == 0
    process.wait.assert_called_once_with(timeout=2)
    handle.timeout = 3
    assert handle.wait(timeout=2) == 0
    process.wait.assert_called_with(timeout=3)


def test_exit_waits_for_running_process(monkeypatch: pytest.MonkeyPatch):
    handle = SubprocessHandle(["command"])
    wait = Mock(return_value=0)
    monkeypatch.setattr(handle, "poll", Mock(return_value=None))
    monkeypatch.setattr(handle, "wait", wait)

    handle.__exit__(None, None, None)

    wait.assert_called_once_with(None)


def test_strip_ansi_sequences():
    assert strip_ansi("before\x1b[31mafter\x1b[0m") == "beforeafter"
