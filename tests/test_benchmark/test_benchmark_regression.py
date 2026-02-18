import json
from pathlib import Path

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


@pytest.mark.parametrize(
    "model_slug",
    _model_slugs("rvc4"),
    ids=[_model_id(s) for s in _model_slugs("rvc4")],
)
def test_benchmark_fps(
    model_slug: str,
    device_ip: str | None,
    benchmark_target: str,
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

    assert fps_min <= actual_fps <= fps_max, (
        f"FPS regression for {model_slug}: "
        f"actual={actual_fps:.2f}, expected={expected_fps:.2f} "
        f"(deviation: {deviation_pct:+.1f}%, "
        f"allowed range: [{fps_min:.2f}, {fps_max:.2f}])"
    )
