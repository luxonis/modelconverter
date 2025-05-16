from modelconverter.packages.base_analyze import Analyzer
from modelconverter.packages.base_benchmark import Benchmark
from modelconverter.packages.base_exporter import Exporter
from modelconverter.packages.base_inferer import Inferer
from modelconverter.packages.base_visualize import Visualizer
from modelconverter.utils.types import Target


def get_exporter(target: Target, *args, **kwargs) -> Exporter:
    if target is Target.RVC2:
        from modelconverter.packages.rvc2.exporter import RVC2Exporter

        return RVC2Exporter(*args, **kwargs)

    if target is Target.RVC3:
        from modelconverter.packages.rvc3.exporter import RVC3Exporter

        return RVC3Exporter(*args, **kwargs)

    if target is Target.RVC4:
        from modelconverter.packages.rvc4.exporter import RVC4Exporter

        return RVC4Exporter(*args, **kwargs)

    if target is Target.HAILO:
        from modelconverter.packages.hailo.exporter import HailoExporter

        return HailoExporter(*args, **kwargs)


def get_inferer(target: Target, *args, **kwargs) -> Inferer:
    if target is Target.RVC2:
        from modelconverter.packages.rvc2.inferer import RVC2Inferer

        return RVC2Inferer.from_config(*args, **kwargs)

    if target is Target.RVC3:
        from modelconverter.packages.rvc3.inferer import RVC3Inferer

        return RVC3Inferer.from_config(*args, **kwargs)

    if target is Target.RVC4:
        from modelconverter.packages.rvc4.inferer import RVC4Inferer

        return RVC4Inferer.from_config(*args, **kwargs)

    if target is Target.HAILO:
        from modelconverter.packages.hailo.inferer import HailoInferer

        return HailoInferer.from_config(*args, **kwargs)


def get_benchmark(target: Target, *args, **kwargs) -> Benchmark:
    if target is Target.RVC2:
        from modelconverter.packages.rvc2.benchmark import RVC2Benchmark

        return RVC2Benchmark(*args, **kwargs)

    if target is Target.RVC3:
        from modelconverter.packages.rvc3.benchmark import RVC3Benchmark

        return RVC3Benchmark(*args, **kwargs)

    if target is Target.RVC4:
        from modelconverter.packages.rvc4.benchmark import RVC4Benchmark

        return RVC4Benchmark(*args, **kwargs)

    if target is Target.HAILO:
        raise NotImplementedError("Hailo Benchmark is not implemented yet.")


def get_analyzer(target: Target, *args, **kwargs) -> Analyzer:
    if target is Target.RVC4:
        from modelconverter.packages.rvc4.analyze import RVC4Analyzer

        return RVC4Analyzer(*args, **kwargs)

    raise ValueError(f"Analyzer not available for {target.name}")


def get_visualizer(target: Target, *args, **kwargs) -> Visualizer:
    if target is Target.RVC4:
        from modelconverter.packages.rvc4.visualize import RVC4Visualizer

        return RVC4Visualizer(*args, **kwargs)
    raise ValueError(f"Visualizer not available for {target.name}")
