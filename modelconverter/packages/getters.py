from typing import Type

from modelconverter.utils.types import Target

from ..packages.base_benchmark import Benchmark
from ..packages.base_exporter import Exporter
from ..packages.base_inferer import Inferer


def get_exporter(target: Target) -> Type[Exporter]:
    if target == Target.RVC2:
        from modelconverter.packages.rvc2.exporter import RVC2Exporter

        return RVC2Exporter

    elif target == Target.RVC3:
        from modelconverter.packages.rvc3.exporter import RVC3Exporter

        return RVC3Exporter

    elif target == Target.RVC4:
        from modelconverter.packages.rvc4.exporter import RVC4Exporter

        return RVC4Exporter

    elif target == Target.HAILO:
        from modelconverter.packages.hailo.exporter import HailoExporter

        return HailoExporter


def get_inferer(target: Target) -> Type[Inferer]:
    if target == Target.RVC2:
        from modelconverter.packages.rvc2.inferer import RVC2Inferer

        return RVC2Inferer

    elif target == Target.RVC3:
        from modelconverter.packages.rvc3.inferer import RVC3Inferer

        return RVC3Inferer

    elif target == Target.RVC4:
        from modelconverter.packages.rvc4.inferer import RVC4Inferer

        return RVC4Inferer

    elif target == Target.HAILO:
        from modelconverter.packages.hailo.inferer import HailoInferer

        return HailoInferer


def get_benchmark(target: Target) -> Type[Benchmark]:
    if target == Target.RVC2:
        from modelconverter.packages.rvc2.benchmark import RVC2Benchmark

        return RVC2Benchmark

    elif target == Target.RVC3:
        from modelconverter.packages.rvc3.benchmark import RVC3Benchmark

        return RVC3Benchmark

    elif target == Target.RVC4:
        from modelconverter.packages.rvc4.benchmark import RVC4Benchmark

        return RVC4Benchmark

    elif target == Target.HAILO:
        from modelconverter.packages.hailo.benchmark import HailoBenchmark

        return HailoBenchmark
