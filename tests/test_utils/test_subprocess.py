import pytest

from modelconverter.utils.exceptions import SubprocessException
from modelconverter.utils.subprocess import subprocess_run


def test_true():
    result = subprocess_run("true")
    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b""


def test_false():
    with pytest.raises(SubprocessException):
        subprocess_run("false")


def test_echo():
    result = subprocess_run("echo hello")
    assert result.returncode == 0
    assert result.stdout == b"hello\n"
    assert result.stderr == b""


def test_stderr():
    result = subprocess_run(
        ["python", "-c", "import sys; sys.stderr.write('hello')"]
    )
    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b"hello"
