from pathlib import Path

import pytest

from modelconverter.platforms.base_benchmark import (
    Benchmark,
    Configuration,
    Result,
    get_option,
    get_optional_option,
)

DEFAULTS: Configuration = {
    "repetitions": 10,
    "benchmark_time": 20,
    "num_threads": 2,
    "device_ip": None,
}


class _FakeBenchmark(Benchmark):
    """A benchmark that records the configurations it received."""

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.seen: list[Configuration] = []

    @property
    def default_configuration(self) -> Configuration:
        return dict(DEFAULTS)

    @property
    def all_configurations(self) -> list[Configuration]:
        return [{"num_threads": threads} for threads in (1, 2)]

    def benchmark(self, configuration: Configuration) -> Result:
        self.seen.append(dict(configuration))
        return {"fps": 1.0, "latency": 1.0}


@pytest.fixture
def model_path(tmp_path: Path) -> str:
    path = tmp_path / "model.blob"
    path.write_bytes(b"")
    return str(path)


def test_full_run_fills_in_the_defaults(model_path: str):
    benchmark = _FakeBenchmark(model_path)

    benchmark.run(full=True, save=False)

    assert [config["num_threads"] for config in benchmark.seen] == [1, 2]
    for config in benchmark.seen:
        assert config["repetitions"] == 10
        assert config["benchmark_time"] == 20
        assert config["device_ip"] is None


def test_explicit_options_win_over_the_defaults(model_path: str):
    benchmark = _FakeBenchmark(model_path)

    benchmark.run(full=False, save=False, repetitions="7")

    assert benchmark.seen == [
        {
            "repetitions": 7,
            "benchmark_time": 20,
            "num_threads": 2,
            "device_ip": None,
        }
    ]


def test_a_varied_option_is_not_overwritten(model_path: str):
    benchmark = _FakeBenchmark(model_path)

    benchmark.run(full=True, save=False, num_threads=8)

    assert [config["num_threads"] for config in benchmark.seen] == [1, 2]


def test_get_option_rejects_a_missing_option():
    with pytest.raises(TypeError, match="must be of type 'int'"):
        get_option({}, "repetitions", int)


def test_get_optional_option_accepts_an_unset_option():
    assert get_optional_option({"device_ip": None}, "device_ip", str) is None
