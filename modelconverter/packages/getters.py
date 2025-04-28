from modelconverter.packages.base_analyze import Analyzer
from modelconverter.packages.base_benchmark import Benchmark
from modelconverter.packages.base_exporter import Exporter
from modelconverter.packages.base_inferer import Inferer
from modelconverter.packages.base_visualize import Visualizer
from modelconverter.utils.types import Target


def get_exporter(target: Target) -> type[Exporter]:
    if target is Target.RVC2:
        from modelconverter.packages.rvc2.exporter import RVC2Exporter

        return RVC2Exporter

    if target is Target.RVC3:
        from modelconverter.packages.rvc3.exporter import RVC3Exporter

        return RVC3Exporter

    if target is Target.RVC4:
        from modelconverter.packages.rvc4.exporter import RVC4Exporter

        return RVC4Exporter

    if target is Target.HAILO:
        from modelconverter.packages.hailo.exporter import HailoExporter

        return HailoExporter


def get_inferer(target: Target) -> type[Inferer]:
    if target is Target.RVC2:
        from modelconverter.packages.rvc2.inferer import RVC2Inferer

        return RVC2Inferer

    if target is Target.RVC3:
        from modelconverter.packages.rvc3.inferer import RVC3Inferer

        return RVC3Inferer

    if target is Target.RVC4:
        from modelconverter.packages.rvc4.inferer import RVC4Inferer

        return RVC4Inferer

    if target is Target.HAILO:
        from modelconverter.packages.hailo.inferer import HailoInferer

        return HailoInferer


def get_benchmark(target: Target) -> type[Benchmark]:
    if target is Target.RVC2:
        from modelconverter.packages.rvc2.benchmark import RVC2Benchmark

        return RVC2Benchmark

    if target is Target.RVC3:
        from modelconverter.packages.rvc3.benchmark import RVC3Benchmark

        return RVC3Benchmark

    if target is Target.RVC4:
        from modelconverter.packages.rvc4.benchmark import RVC4Benchmark

        return RVC4Benchmark

    if target is Target.HAILO:
        from modelconverter.packages.hailo.benchmark import HailoBenchmark

        return HailoBenchmark


def get_analyzer(target: Target) -> type[Analyzer]:
    if target is Target.RVC4:
        from modelconverter.packages.rvc4.analyze import RVC4Analyzer

        return RVC4Analyzer

    raise ValueError(f"Analyzer not available for {target.name}")


def get_visualizer(target: Target) -> type[Visualizer]:
    if target is Target.RVC4:
        from modelconverter.packages.rvc4.visualize import RVC4Visualizer

        return RVC4Visualizer
    raise ValueError(f"Visualizer not available for {target.name}")
