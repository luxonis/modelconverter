"""Factories for the platform-specific conversion components.

Every conversion platform (RVC2, RVC3, RVC4 and Hailo) provides its own
exporter and inferer; the benchmark is missing for Hailo, and the
analyzer and the visualizer are provided by RVC4 only. The getters in
this module map a `Platform` to the matching implementation and import
it lazily, so that only the dependencies of the selected component
need to be available. The exporter and inferer dependencies live in
the platform's Docker image; the host-run benchmark, analyzer and
visualizer take theirs from the host environment.
"""

from modelconverter.platforms.base_analyze import Analyzer
from modelconverter.platforms.base_benchmark import Benchmark
from modelconverter.platforms.base_exporter import Exporter
from modelconverter.platforms.base_inferer import Inferer
from modelconverter.platforms.base_visualize import Visualizer
from modelconverter.utils.types import Platform


def get_exporter(platform: Platform, *args, **kwargs) -> Exporter:
    """Create the `Exporter` implementation for the given platform.

    Args:
        platform: Platform the model is converted for.
        *args: Positional arguments passed to the exporter constructor.
        **kwargs: Keyword arguments passed to the exporter constructor.

    Returns:
        Exporter for the given platform.

    """
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
    """Create the `Inferer` implementation for the given platform.

    The inferer is constructed using its ``from_config`` constructor.

    Args:
        platform: Platform the model was converted for.
        *args: Positional arguments passed to ``from_config``.
        **kwargs: Keyword arguments passed to ``from_config``.

    Returns:
        Inferer for the given platform.

    """
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
    """Create the `Benchmark` implementation for the given platform.

    Args:
        platform: Platform to benchmark the model on.
        *args: Positional arguments passed to the benchmark constructor.
        **kwargs: Keyword arguments passed to the benchmark constructor.

    Returns:
        Benchmark for the given platform.

    Raises:
        NotImplementedError: If ``platform`` is ``Platform.HAILO``.

    """
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
    """Create the `Analyzer` implementation for the given platform.

    Only RVC4 provides an analyzer.

    Args:
        platform: Platform the model was converted for.
        *args: Positional arguments passed to the analyzer constructor.
        **kwargs: Keyword arguments passed to the analyzer constructor.

    Returns:
        Analyzer for the given platform.

    Raises:
        ValueError: If ``platform`` is not ``Platform.RVC4``.

    """
    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.analyze import RVC4Analyzer

        return RVC4Analyzer(*args, **kwargs)

    raise ValueError(f"Analyzer not available for {platform.name}")


def get_visualizer(platform: Platform, *args, **kwargs) -> Visualizer:
    """Create the `Visualizer` implementation for the given platform.

    Only RVC4 provides a visualizer.

    Args:
        platform: Platform the model was converted for.
        *args: Positional arguments passed to the visualizer
            constructor.
        **kwargs: Keyword arguments passed to the visualizer
            constructor.

    Returns:
        Visualizer for the given platform.

    Raises:
        ValueError: If ``platform`` is not ``Platform.RVC4``.

    """
    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.visualize import RVC4Visualizer

        return RVC4Visualizer(*args, **kwargs)
    raise ValueError(f"Visualizer not available for {platform.name}")
