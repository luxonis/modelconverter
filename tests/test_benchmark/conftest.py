import os
from datetime import datetime, timezone
from uuid import uuid4

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
    parser.addoption(
        "--testbed-name",
        action="store",
        default=None,
        help="Logical HIL testbed name for Influx metadata.",
    )
    parser.addoption(
        "--camera-mxid",
        action="store",
        default=None,
        help="Camera MXID for Influx metadata.",
    )
    parser.addoption(
        "--camera-os-version",
        action="store",
        default=None,
        help="Camera OS version for Influx metadata.",
    )
    parser.addoption(
        "--camera-model",
        action="store",
        default=None,
        help="Camera model for Influx metadata.",
    )
    parser.addoption(
        "--camera-agent-version",
        action="store",
        default=None,
        help="Camera agent version for Influx metadata.",
    )
    parser.addoption(
        "--runner",
        action="store",
        default=None,
        help="Runner name for Influx metadata.",
    )
    parser.addoption(
        "--server-os",
        action="store",
        default=None,
        help="Server OS for Influx metadata.",
    )
    parser.addoption(
        "--run-id",
        action="store",
        default=None,
        help="Optional run identifier for grouping benchmark results in Influx.",
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


def _option_or_env(
    request: pytest.FixtureRequest,
    option_name: str,
    env_name: str,
) -> str | None:
    return request.config.getoption(option_name) or os.environ.get(env_name)


@pytest.fixture(scope="session")
def influx_metadata(request: pytest.FixtureRequest) -> dict[str, str | None]:
    return {
        "testbed_name": _option_or_env(
            request, "--testbed-name", "HIL_TESTBED_NAME"
        ),
        "camera_mxid": _option_or_env(
            request, "--camera-mxid", "HIL_CAMERA_MXID"
        ),
        "camera_os_version": _option_or_env(
            request, "--camera-os-version", "HIL_CAMERA_OS_VERSION"
        ),
        "camera_model": _option_or_env(
            request, "--camera-model", "HIL_CAMERA_MODEL"
        ),
        "camera_agent_version": _option_or_env(
            request, "--camera-agent-version", "HIL_CAMERA_AGENT_VERSION"
        ),
        "runner": _option_or_env(request, "--runner", "HIL_RUNNER")
        or os.environ.get("GITHUB_RUNNER_NAME")
        or os.environ.get("HOSTNAME")
        or os.environ.get("USER"),
        "server_os": _option_or_env(request, "--server-os", "HIL_SERVER_OS"),
    }


@pytest.fixture(scope="session")
def benchmark_run_id(request: pytest.FixtureRequest) -> str:
    configured_run_id = (
        request.config.getoption("--run-id") or os.environ.get("HIL_RUN_ID")
    )
    if configured_run_id:
        return configured_run_id

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"benchmark-{timestamp}-{uuid4().hex[:8]}"
