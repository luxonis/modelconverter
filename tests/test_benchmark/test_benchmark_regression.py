import json
from pathlib import Path
from typing import Any

import pytest

from hil_framework.lib_testbed.db_source.InfluxClient import InfluxClient
from modelconverter.packages import get_benchmark
from modelconverter.utils.types import Target

TARGETS_FILE = Path(__file__).parent / "benchmark_targets.json"

_targets_data = json.loads(TARGETS_FILE.read_text())


def _model_slugs(target: str) -> list[str]:
    return list(_targets_data.get(target, {}).keys())


def _model_id(slug: str) -> str:
    """Take out the `luxonis` and use the remainder of the slug to name
    the test."""
    return slug.rsplit("/", 1)[-1]


def _build_fps_benchmark_data(
    *,
    bucket: str,
    token: str,
    model_slug: str,
    benchmark_target: str,
    benchmark_run_id: str,
    device_ip: str | None,
    actual_fps: float,
    expected_fps: float,
    tolerance_low: float,
    tolerance_high: float,
    fps_min: float,
    fps_max: float,
    deviation_pct: float,
    success: bool,
    influx_metadata: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "name": "fps_benchmark",
        "bucket": bucket,
        "token": token,
        "tags": [
            {"name": "model_slug", "value": model_slug},
            {"name": "benchmark_target", "value": benchmark_target},
            {"name": "run_id", "value": benchmark_run_id},
            {"name": "status", "value": "passed" if success else "failed"},
            {
                "name": "testbed_name",
                "value": influx_metadata.get("testbed_name"),
            },
            {
                "name": "camera_mxid",
                "value": influx_metadata.get("camera_mxid"),
            },
            {
                "name": "camera_os_version",
                "value": influx_metadata.get("camera_os_version"),
            },
            {
                "name": "camera_model",
                "value": influx_metadata.get("camera_model"),
            },
            {
                "name": "camera_revision",
                "value": influx_metadata.get("camera_revision"),
            },
            {
                "name": "server_os",
                "value": influx_metadata.get("server_os"),
            },
            {
                "name": "depthai_version",
                "value": influx_metadata.get("depthai_version"),
            },
            {"name": "device_ip", "value": device_ip},
        ],
        "fields": [
            {"name": "actual_fps", "value": actual_fps},
            {"name": "expected_fps", "value": expected_fps},
            {"name": "tolerance_low", "value": tolerance_low},
            {"name": "tolerance_high", "value": tolerance_high},
            {"name": "fps_min", "value": fps_min},
            {"name": "fps_max", "value": fps_max},
            {"name": "deviation_pct", "value": deviation_pct},
            {"name": "success", "value": success},
        ],
    }


def _write_fps_benchmark_result(
    *,
    bucket: str,
    token: str,
    model_slug: str,
    benchmark_target: str,
    benchmark_run_id: str,
    device_ip: str | None,
    actual_fps: float,
    expected_fps: float,
    tolerance_low: float,
    tolerance_high: float,
    fps_min: float,
    fps_max: float,
    deviation_pct: float,
    success: bool,
    influx_metadata: dict[str, str | None],
) -> None:
    benchmark_data = _build_fps_benchmark_data(
        bucket=bucket,
        token=token,
        model_slug=model_slug,
        benchmark_target=benchmark_target,
        benchmark_run_id=benchmark_run_id,
        device_ip=device_ip,
        actual_fps=actual_fps,
        expected_fps=expected_fps,
        tolerance_low=tolerance_low,
        tolerance_high=tolerance_high,
        fps_min=fps_min,
        fps_max=fps_max,
        deviation_pct=deviation_pct,
        success=success,
        influx_metadata=influx_metadata,
    )
    client = InfluxClient(bucket, token=token)
    print(
        "Writing fps_benchmark point to InfluxDB: "
        f"bucket={client.INFLUXDB_BUCKET}, org={client.INFLUXDB_ORG}"
    )
    try:
        client.save_benchmark_data(benchmark_data)
    finally:
        client.close()


@pytest.mark.parametrize(
    "model_slug",
    _model_slugs("rvc4"),
    ids=[_model_id(s) for s in _model_slugs("rvc4")],
)
def test_benchmark_fps(
    model_slug: str,
    device_ip: str | None,
    benchmark_target: str,
    benchmark_run_id: str,
    influx_bucket: str | None,
    influx_token: str | None,
    influx_metadata: dict[str, str | None],
) -> None:
    model_config = _targets_data[benchmark_target][model_slug]
    expected_fps = model_config["expected_fps"]

    if expected_fps is None:
        pytest.skip(
            f"No expected_fps set for {model_slug}."
            "Establish a baseline and add it to benchmark_targets.json."
        )

    target_enum = Target(benchmark_target)

    bench = get_benchmark(target_enum, model_slug)
    configuration = {
        **bench.default_configuration,
        "power_benchmark": False,
        "dsp_benchmark": False,
    }
    if device_ip is not None:
        configuration["device_ip"] = device_ip

    result = bench.benchmark(configuration)
    actual_fps = result.fps

    tolerance_low = model_config["tolerance_low"]
    tolerance_high = model_config["tolerance_high"]
    fps_min = expected_fps * (1 - tolerance_low)
    fps_max = expected_fps * (1 + tolerance_high)

    deviation_pct = ((actual_fps - expected_fps) / expected_fps) * 100
    success = fps_min <= actual_fps <= fps_max

    print(
        f"Benchmark result for {model_slug}: "
        f"actual={actual_fps:.2f} FPS, expected={expected_fps:.2f} FPS. "
    )

    _write_fps_benchmark_result(
        bucket=influx_bucket,
        token=influx_token,
        model_slug=model_slug,
        benchmark_target=benchmark_target,
        benchmark_run_id=benchmark_run_id,
        device_ip=device_ip,
        actual_fps=actual_fps,
        expected_fps=expected_fps,
        tolerance_low=tolerance_low,
        tolerance_high=tolerance_high,
        fps_min=fps_min,
        fps_max=fps_max,
        deviation_pct=deviation_pct,
        success=success,
        influx_metadata=influx_metadata,
    )

    assert success, (
        f"FPS regression for {model_slug}: "
        f"actual={actual_fps:.2f}, expected={expected_fps:.2f} "
        f"(deviation: {deviation_pct:+.1f}%, "
        f"allowed range: [{fps_min:.2f}, {fps_max:.2f}])"
    )
