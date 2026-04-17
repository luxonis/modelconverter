import os

import pytest
from hil_framework.lib_testbed.utils.Testbed import Testbed

INFLUX_BUCKET = "fps_metrics"


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
    parser.addoption(
        "--benchmark-run-id",
        action="store",
        default=None,
        help="Benchmark run identifier to store in InfluxDB.",
    )
    parser.addoption(
        "--depthai-version",
        action="store",
        default=None,
        help="DepthAI version recorded in benchmark metadata.",
    )
    parser.addoption(
        "--testbed-name",
        action="store",
        default=None,
        help="Testbed name recorded in benchmark metadata.",
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


@pytest.fixture(scope="session")
def depthai_version(request: pytest.FixtureRequest) -> str | None:
    return request.config.getoption("--depthai-version")


@pytest.fixture(scope="session")
def testbed_name(request: pytest.FixtureRequest) -> str | None:
    configured_testbed_name = request.config.getoption("--testbed-name")
    if configured_testbed_name:
        return configured_testbed_name
    return os.environ.get("HIL_TESTBED")


@pytest.fixture(scope="session")
def hil_testbed(testbed_name: str | None) -> Testbed:
    if not testbed_name:
        pytest.exit(
            "HIL_TESTBED environment variable or --testbed-name must be set.",
            returncode=1,
        )
    return Testbed(testbed_name)


@pytest.fixture(scope="session")
def benchmark_run_id(request: pytest.FixtureRequest) -> str:
    configured_run_id = request.config.getoption("--benchmark-run-id")
    if configured_run_id:
        return configured_run_id

    from datetime import datetime, timezone
    from uuid import uuid4

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"benchmark-{timestamp}-{uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def influx_bucket() -> str:
    return INFLUX_BUCKET


@pytest.fixture(scope="session")
def influx_token() -> str | None:
    return os.environ.get("INFLUX_TOKEN")
