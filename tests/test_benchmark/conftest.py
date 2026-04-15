import os

import pytest

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
    parser.addoption(
        "--camera-mxid",
        action="store",
        default=None,
        help="Camera MXID recorded in benchmark metadata.",
    )
    parser.addoption(
        "--camera-os-version",
        action="store",
        default=None,
        help="Camera OS version recorded in benchmark metadata.",
    )
    parser.addoption(
        "--camera-model",
        action="store",
        default=None,
        help="Camera model recorded in benchmark metadata.",
    )
    parser.addoption(
        "--camera-revision",
        action="store",
        default=None,
        help="Camera revision recorded in benchmark metadata.",
    )
    parser.addoption(
        "--server-os",
        action="store",
        default=None,
        help="Server OS recorded in benchmark metadata.",
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
def influx_metadata(request: pytest.FixtureRequest) -> dict[str, str | None]:
    return {
        "testbed_name": request.config.getoption("--testbed-name"),
        "camera_mxid": request.config.getoption("--camera-mxid"),
        "camera_os_version": request.config.getoption("--camera-os-version"),
        "camera_model": request.config.getoption("--camera-model"),
        "camera_revision": request.config.getoption("--camera-revision"),
        "server_os": request.config.getoption("--server-os"),
        "depthai_version": request.config.getoption("--depthai-version"),
    }


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
