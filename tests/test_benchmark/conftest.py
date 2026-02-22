import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device-ip",
        action="store",
        default=None,
        help="IP address of the target device.",
    )
    parser.addoption(
        "--benchmark-target",
        action="store",
        default="rvc4",
        help="Target platform to benchmark (default: rvc4).",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not os.environ.get("HUBAI_API_KEY"):
        pytest.exit(
            "HUBAI_API_KEY environment variable is not set.",
            returncode=1,
        )


@pytest.fixture
def device_ip(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--device-ip")


@pytest.fixture
def benchmark_target(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--benchmark-target")
