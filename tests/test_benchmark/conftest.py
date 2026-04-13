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


@pytest.fixture(scope="session")
def influx_metadata(request: pytest.FixtureRequest) -> dict[str, str | None]:
    return {
        "testbed_name": os.environ.get("HIL_TESTBED"),
        "camera_mxid": os.environ.get("HIL_CAMERA_MXID"),
        "camera_os_version": os.environ.get("HIL_CAMERA_OS_VERSION"),
        "camera_model": os.environ.get("HIL_CAMERA_MODEL"),
        "camera_revision": os.environ.get("HIL_CAMERA_REVISION"),
        "runner": os.environ.get("HIL_RUNNER")
        or os.environ.get("GITHUB_RUNNER_NAME")
        or os.environ.get("HOSTNAME")
        or os.environ.get("USER"),
        "server_os": os.environ.get("HIL_SERVER_OS"),
        "depthai_version": os.environ.get("DEPTHAI_VERSION"),
    }


@pytest.fixture(scope="session")
def benchmark_run_id(request: pytest.FixtureRequest) -> str:
    configured_run_id = os.environ.get("HIL_RUN_ID")
    if configured_run_id:
        return configured_run_id

    from datetime import datetime, timezone
    from uuid import uuid4

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"benchmark-{timestamp}-{uuid4().hex[:8]}"
