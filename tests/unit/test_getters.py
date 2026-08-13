"""Host-side unit tests for ``modelconverter.platforms.getters``.

``getters`` is pure dispatch: each factory lazily imports a backend class and
constructs it. Those backends live in vendor-heavy modules (RVC2/RVC3 need
``tflite2onnx``/``openvino``, Hailo needs ``hailo_sdk_client``), so the success
branches are covered by injecting a fake backend module into ``sys.modules`` at
the path the factory imports from. The RVC4 exporter imports cleanly host-side,
so it also gets a real-construction smoke test.
"""

import sys
import types
from pathlib import Path

import pytest

from modelconverter.platforms import getters
from modelconverter.utils.types import Platform
from tests.helpers.onnx_factory import single_io_onnx


class _Recorder:
    """A stand-in backend that records how it was instantiated."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.via_from_config = False

    @classmethod
    def from_config(cls, *args, **kwargs) -> "_Recorder":
        inst = cls(*args, **kwargs)
        inst.via_from_config = True
        return inst


def _inject_fake_backend(
    monkeypatch: pytest.MonkeyPatch, module_path: str, class_name: str
) -> None:
    """Register a fake backend module so ``from <path> import <cls>``
    resolves.

    The real subpackage ``__init__`` files are empty, so overriding only
    the leaf module in ``sys.modules`` is sufficient and is undone after
    the test by ``monkeypatch``.
    """
    module = types.ModuleType(module_path)
    setattr(module, class_name, _Recorder)
    monkeypatch.setitem(sys.modules, module_path, module)


GET_EXPORTER_FAKE_CASES = [
    (Platform.RVC2, "modelconverter.platforms.rvc2.exporter", "RVC2Exporter"),
    (Platform.RVC3, "modelconverter.platforms.rvc3.exporter", "RVC3Exporter"),
    (Platform.RVC4, "modelconverter.platforms.rvc4.exporter", "RVC4Exporter"),
    (
        Platform.HAILO,
        "modelconverter.platforms.hailo.exporter",
        "HailoExporter",
    ),
]


@pytest.mark.parametrize(
    ("platform", "path", "cls_name"), GET_EXPORTER_FAKE_CASES
)
def test_get_exporter_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    platform: Platform,
    path: str,
    cls_name: str,
):
    _inject_fake_backend(monkeypatch, path, cls_name)
    exporter = getters.get_exporter(platform, "cfg", "out_dir")
    assert isinstance(exporter, _Recorder)
    assert exporter.args == ("cfg", "out_dir")


def test_get_exporter_rvc4_real_construction(work_dir: Path):
    """Smoke test the RVC4 branch against the real exporter class."""
    from modelconverter.platforms.rvc4.exporter import RVC4Exporter
    from modelconverter.utils.config import Config

    model = single_io_onnx(work_dir / "m.onnx").resolve()
    cfg = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": [1, 3, 64, 64],
            "rvc4.disable_calibration": True,
        },
    )
    stage = next(iter(cfg.stages.values()))
    output_dir = work_dir / "out"
    output_dir.mkdir()
    exporter = getters.get_exporter(Platform.RVC4, stage, output_dir)
    assert isinstance(exporter, RVC4Exporter)


GET_INFERER_FAKE_CASES = [
    (Platform.RVC2, "modelconverter.platforms.rvc2.inferer", "RVC2Inferer"),
    (Platform.RVC3, "modelconverter.platforms.rvc3.inferer", "RVC3Inferer"),
    (Platform.RVC4, "modelconverter.platforms.rvc4.inferer", "RVC4Inferer"),
    (
        Platform.HAILO,
        "modelconverter.platforms.hailo.inferer",
        "HailoInferer",
    ),
]


@pytest.mark.parametrize(
    ("platform", "path", "cls_name"), GET_INFERER_FAKE_CASES
)
def test_get_inferer_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    platform: Platform,
    path: str,
    cls_name: str,
):
    _inject_fake_backend(monkeypatch, path, cls_name)
    inferer = getters.get_inferer(platform, "cfg")
    assert isinstance(inferer, _Recorder)
    # Inferers are built through the ``from_config`` classmethod.
    assert inferer.via_from_config


GET_BENCHMARK_FAKE_CASES = [
    (
        Platform.RVC2,
        "modelconverter.platforms.rvc2.benchmark",
        "RVC2Benchmark",
    ),
    (
        Platform.RVC3,
        "modelconverter.platforms.rvc3.benchmark",
        "RVC3Benchmark",
    ),
    (
        Platform.RVC4,
        "modelconverter.platforms.rvc4.benchmark",
        "RVC4Benchmark",
    ),
]


@pytest.mark.parametrize(
    ("platform", "path", "cls_name"), GET_BENCHMARK_FAKE_CASES
)
def test_get_benchmark_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    platform: Platform,
    path: str,
    cls_name: str,
):
    _inject_fake_backend(monkeypatch, path, cls_name)
    benchmark = getters.get_benchmark(platform, "model")
    assert isinstance(benchmark, _Recorder)
    assert benchmark.args == ("model",)


def test_get_benchmark_hailo_not_implemented():
    with pytest.raises(NotImplementedError, match="Hailo Benchmark"):
        getters.get_benchmark(Platform.HAILO)


def test_get_analyzer_rvc4_dispatch(monkeypatch: pytest.MonkeyPatch):
    _inject_fake_backend(
        monkeypatch,
        "modelconverter.platforms.rvc4.analyze",
        "RVC4Analyzer",
    )
    analyzer = getters.get_analyzer(Platform.RVC4, "arg")
    assert isinstance(analyzer, _Recorder)


@pytest.mark.parametrize(
    "platform", [Platform.RVC2, Platform.RVC3, Platform.HAILO]
)
def test_get_analyzer_unsupported_target(platform: Platform):
    with pytest.raises(ValueError, match="Analyzer not available"):
        getters.get_analyzer(platform)


def test_get_visualizer_rvc4_dispatch(monkeypatch: pytest.MonkeyPatch):
    _inject_fake_backend(
        monkeypatch,
        "modelconverter.platforms.rvc4.visualize",
        "RVC4Visualizer",
    )
    visualizer = getters.get_visualizer(Platform.RVC4, "arg")
    assert isinstance(visualizer, _Recorder)


@pytest.mark.parametrize(
    "platform", [Platform.RVC2, Platform.RVC3, Platform.HAILO]
)
def test_get_visualizer_unsupported_target(platform: Platform):
    with pytest.raises(ValueError, match="Visualizer not available"):
        getters.get_visualizer(platform)
