"""Regression coverage for RVC4 benchmark device selection."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modelconverter.platforms.rvc4 import benchmark as rvc4_benchmark


class _FakeMonitor:
    def __init__(self, handler: object) -> None:
        self.handler = handler

    def get_idle_measurements(self) -> dict[str, float | None]:
        return {}

    def get_stats(self) -> dict[str, float | None]:
        return {}

    def stop(self) -> None:
        pass


def _benchmark() -> rvc4_benchmark.RVC4Benchmark:
    benchmark = rvc4_benchmark.RVC4Benchmark.__new__(
        rvc4_benchmark.RVC4Benchmark
    )
    benchmark.model_path = Path("model.tar.xz")
    benchmark.model_name = "model"
    return benchmark


def _configuration(*, device_monitor: bool) -> rvc4_benchmark.Configuration:
    config = _benchmark().default_configuration
    config["device_monitor"] = device_monitor
    return config


def _rvc4_info(name: str, device_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        platform=rvc4_benchmark.XLinkPlatform.X_LINK_RVC4,
        getDeviceId=lambda: device_id,
    )


def test_monitored_dai_auto_selection_uses_same_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rvc4_benchmark.dai.Device,
        "getAllAvailableDevices",
        lambda: [
            _rvc4_info("192.0.2.20", "1002"),
            _rvc4_info("192.0.2.21", "1003"),
        ],
    )
    create_handler = Mock(return_value=object())
    benchmark_device_ips: list[str | None] = []

    def benchmark_dai(
        self: rvc4_benchmark.RVC4Benchmark,
        *args: object,
        device_ip: str | None = None,
        **kwargs: object,
    ) -> rvc4_benchmark.Result:
        benchmark_device_ips.append(device_ip)
        assert self._monitor is not None
        return {"fps": 1.0, "latency": "N/A"}

    monkeypatch.setattr(rvc4_benchmark, "create_handler", create_handler)
    monkeypatch.setattr(rvc4_benchmark, "DeviceMonitor", _FakeMonitor)
    monkeypatch.setattr(
        rvc4_benchmark.RVC4Benchmark, "_benchmark_dai", benchmark_dai
    )

    result = _benchmark().benchmark(_configuration(device_monitor=True))

    create_handler.assert_called_once_with("192.0.2.20", "3ea")
    assert benchmark_device_ips == ["192.0.2.20"]
    assert result == {"fps": 1.0, "latency": "N/A"}


def test_unmonitored_dai_keeps_depthai_auto_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enumerate_devices = Mock(
        side_effect=AssertionError(
            "unmonitored DAI benchmark should not resolve a device early"
        )
    )
    create_handler = Mock()
    benchmark_device_ips: list[str | None] = []

    def benchmark_dai(
        self: rvc4_benchmark.RVC4Benchmark,
        *args: object,
        device_ip: str | None = None,
        **kwargs: object,
    ) -> rvc4_benchmark.Result:
        benchmark_device_ips.append(device_ip)
        assert self._handler is None
        return {"fps": 1.0, "latency": "N/A"}

    monkeypatch.setattr(
        rvc4_benchmark.dai.Device,
        "getAllAvailableDevices",
        enumerate_devices,
    )
    monkeypatch.setattr(rvc4_benchmark, "create_handler", create_handler)
    monkeypatch.setattr(
        rvc4_benchmark.RVC4Benchmark, "_benchmark_dai", benchmark_dai
    )

    result = _benchmark().benchmark(_configuration(device_monitor=False))

    assert result == {"fps": 1.0, "latency": "N/A"}
    create_handler.assert_not_called()
    enumerate_devices.assert_not_called()
    assert benchmark_device_ips == [None]
