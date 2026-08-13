from modelconverter.platforms.base_analyze import Analyzer
from modelconverter.platforms.base_benchmark import Benchmark
from modelconverter.platforms.base_exporter import Exporter
from modelconverter.platforms.base_inferer import Inferer
from modelconverter.platforms.base_visualize import Visualizer
from modelconverter.utils.types import Platform


def get_exporter(platform: Platform, *args, **kwargs) -> Exporter:
    if platform is Platform.RVC2:
        from modelconverter.platforms.rvc2.exporter import RVC2Exporter

        return RVC2Exporter(*args, **kwargs)

    if platform is Platform.RVC3:
        from modelconverter.platforms.rvc3.exporter import RVC3Exporter

        return RVC3Exporter(*args, **kwargs)

    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.exporter import RVC4Exporter

        return RVC4Exporter(*args, **kwargs)

    if platform is Platform.HAILO:  # pragma: no branch
        from modelconverter.platforms.hailo.exporter import HailoExporter

        return HailoExporter(*args, **kwargs)


def get_inferer(platform: Platform, *args, **kwargs) -> Inferer:
    if platform is Platform.RVC2:
        from modelconverter.platforms.rvc2.inferer import RVC2Inferer

        return RVC2Inferer.from_config(*args, **kwargs)

    if platform is Platform.RVC3:
        from modelconverter.platforms.rvc3.inferer import RVC3Inferer

        return RVC3Inferer.from_config(*args, **kwargs)

    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.inferer import RVC4Inferer

        return RVC4Inferer.from_config(*args, **kwargs)

    if platform is Platform.HAILO:  # pragma: no branch
        from modelconverter.platforms.hailo.inferer import HailoInferer

        return HailoInferer.from_config(*args, **kwargs)


def get_benchmark(platform: Platform, *args, **kwargs) -> Benchmark:
    if platform is Platform.RVC2:
        from modelconverter.platforms.rvc2.benchmark import RVC2Benchmark

        return RVC2Benchmark(*args, **kwargs)

    if platform is Platform.RVC3:
        from modelconverter.platforms.rvc3.benchmark import RVC3Benchmark

        return RVC3Benchmark(*args, **kwargs)

    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.benchmark import RVC4Benchmark

        return RVC4Benchmark(*args, **kwargs)

    if platform is Platform.HAILO:  # pragma: no branch
        raise NotImplementedError("Hailo Benchmark is not implemented yet.")


def get_analyzer(platform: Platform, *args, **kwargs) -> Analyzer:
    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.analyze import RVC4Analyzer

        return RVC4Analyzer(*args, **kwargs)

    raise ValueError(f"Analyzer not available for {platform.name}")


def get_visualizer(platform: Platform, *args, **kwargs) -> Visualizer:
    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.visualize import RVC4Visualizer

        return RVC4Visualizer(*args, **kwargs)
    raise ValueError(f"Visualizer not available for {platform.name}")
