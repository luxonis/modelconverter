import json
import os
from pathlib import Path
from urllib import error, parse, request

import pytest

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


def _escape_tag(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace(" ", "\\ ")
        .replace("=", "\\=")
    )


def _format_field(value: str | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value}i"
    if isinstance(value, float):
        return repr(value)

    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_fps_result_to_influx(
    *,
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
    influx_url = os.environ.get("INFLUX_HOST")
    influx_org = os.environ.get("INFLUX_ORG")
    influx_bucket = os.environ.get("INFLUX_BUCKET")
    influx_token = os.environ.get("INFLUX_TOKEN")

    if not all([influx_url, influx_org, influx_bucket, influx_token]):
        return

    tags = {
        "model_slug": model_slug,
        "benchmark_target": benchmark_target,
        "run_id": benchmark_run_id,
        "status": "passed" if success else "failed",
    }
    optional_tags = {
        "testbed_name": influx_metadata.get("testbed_name"),
        "camera_mxid": influx_metadata.get("camera_mxid"),
        "camera_os_version": influx_metadata.get("camera_os_version"),
        "camera_model": influx_metadata.get("camera_model"),
        "camera_agent_version": influx_metadata.get("camera_agent_version"),
        "runner": influx_metadata.get("runner"),
        "server_os": influx_metadata.get("server_os"),
        "device_ip": device_ip,
    }
    tags.update(
        {
            key: value
            for key, value in optional_tags.items()
            if value not in (None, "")
        }
    )

    fields = {
        "actual_fps": actual_fps,
        "expected_fps": expected_fps,
        "tolerance_low": tolerance_low,
        "tolerance_high": tolerance_high,
        "fps_min": fps_min,
        "fps_max": fps_max,
        "deviation_pct": deviation_pct,
        "success": success,
    }

    tag_set = ",".join(f"{key}={_escape_tag(value)}" for key, value in tags.items())
    field_set = ",".join(
        f"{key}={_format_field(value)}" for key, value in fields.items()
    )
    line = f"fps_benchmark,{tag_set} {field_set}"

    write_url = (
        f"{influx_url.rstrip('/')}/api/v2/write?"
        f"org={parse.quote(influx_org, safe='')}&"
        f"bucket={parse.quote(influx_bucket, safe='')}&precision=ns"
    )
    influx_request = request.Request(
        write_url,
        data=line.encode(),
        headers={
            "Authorization": f"Token {influx_token}",
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(influx_request, timeout=5):
            return
    except (error.URLError, TimeoutError, OSError) as exc:
        print(f"Failed to write benchmark result to InfluxDB: {exc}")


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

    _write_fps_result_to_influx(
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
