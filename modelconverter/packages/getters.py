"""Factories for the target-specific conversion components.

Every conversion target (RVC2, RVC3, RVC4 and Hailo) provides its own
exporter and inferer; the benchmark is missing for Hailo, and the
analyzer and the visualizer are provided by RVC4 only. The getters in
this module map a `Target` to the matching implementation and import it
lazily, so that only the dependencies of the selected target -- which
are installed in that target's Docker image -- need to be available.
"""

from modelconverter.packages.base_analyze import Analyzer
from modelconverter.packages.base_benchmark import Benchmark
from modelconverter.packages.base_exporter import Exporter
from modelconverter.packages.base_inferer import Inferer
from modelconverter.packages.base_visualize import Visualizer
from modelconverter.utils.types import Target


def get_exporter(target: Target, *args, **kwargs) -> Exporter:
    """Create the `Exporter` implementation for the given target.

    Args:
        target: Target the model is converted for.
        *args: Positional arguments passed to the exporter constructor.
        **kwargs: Keyword arguments passed to the exporter constructor.

    Returns:
        Exporter for the given target.

    """
    if target is Target.RVC2:
        from modelconverter.packages.rvc2.exporter import RVC2Exporter

        return RVC2Exporter(*args, **kwargs)

    if target is Target.RVC3:
        from modelconverter.packages.rvc3.exporter import RVC3Exporter

        return RVC3Exporter(*args, **kwargs)

    if target is Target.RVC4:
        from modelconverter.packages.rvc4.exporter import RVC4Exporter

        return RVC4Exporter(*args, **kwargs)

    if target is Target.HAILO:  # pragma: no branch
        from modelconverter.packages.hailo.exporter import HailoExporter

        return HailoExporter(*args, **kwargs)


def get_inferer(target: Target, *args, **kwargs) -> Inferer:
    """Create the `Inferer` implementation for the given target.

    The inferer is constructed using its ``from_config`` constructor.

    Args:
        target: Target the model was converted for.
        *args: Positional arguments passed to ``from_config``.
        **kwargs: Keyword arguments passed to ``from_config``.

    Returns:
        Inferer for the given target.

    """
    if target is Target.RVC2:
        from modelconverter.packages.rvc2.inferer import RVC2Inferer

        return RVC2Inferer.from_config(*args, **kwargs)

    if target is Target.RVC3:
        from modelconverter.packages.rvc3.inferer import RVC3Inferer

        return RVC3Inferer.from_config(*args, **kwargs)

    if target is Target.RVC4:
        from modelconverter.packages.rvc4.inferer import RVC4Inferer

        return RVC4Inferer.from_config(*args, **kwargs)

    if target is Target.HAILO:  # pragma: no branch
        from modelconverter.packages.hailo.inferer import HailoInferer

        return HailoInferer.from_config(*args, **kwargs)


def get_benchmark(target: Target, *args, **kwargs) -> Benchmark:
    """Create the `Benchmark` implementation for the given target.

    Args:
        target: Target to benchmark the model on.
        *args: Positional arguments passed to the benchmark constructor.
        **kwargs: Keyword arguments passed to the benchmark constructor.

    Returns:
        Benchmark for the given target.

    Raises:
        NotImplementedError: If ``target`` is ``Target.HAILO``.

    """
    if target is Target.RVC2:
        from modelconverter.packages.rvc2.benchmark import RVC2Benchmark

        return RVC2Benchmark(*args, **kwargs)

    if target is Target.RVC3:
        from modelconverter.packages.rvc3.benchmark import RVC3Benchmark

        return RVC3Benchmark(*args, **kwargs)

    if target is Target.RVC4:
        from modelconverter.packages.rvc4.benchmark import RVC4Benchmark

        return RVC4Benchmark(*args, **kwargs)

    if target is Target.HAILO:  # pragma: no branch
        raise NotImplementedError("Hailo Benchmark is not implemented yet.")


def get_analyzer(target: Target, *args, **kwargs) -> Analyzer:
    """Create the `Analyzer` implementation for the given target.

    Only RVC4 provides an analyzer.

    Args:
        target: Target the model was converted for.
        *args: Positional arguments passed to the analyzer constructor.
        **kwargs: Keyword arguments passed to the analyzer constructor.

    Returns:
        Analyzer for the given target.

    Raises:
        ValueError: If ``target`` is not ``Target.RVC4``.

    """
    if target is Target.RVC4:
        from modelconverter.packages.rvc4.analyze import RVC4Analyzer

        return RVC4Analyzer(*args, **kwargs)

    raise ValueError(f"Analyzer not available for {target.name}")


def get_visualizer(target: Target, *args, **kwargs) -> Visualizer:
    """Create the `Visualizer` implementation for the given target.

    Only RVC4 provides a visualizer.

    Args:
        target: Target the model was converted for.
        *args: Positional arguments passed to the visualizer
            constructor.
        **kwargs: Keyword arguments passed to the visualizer
            constructor.

    Returns:
        Visualizer for the given target.

    Raises:
        ValueError: If ``target`` is not ``Target.RVC4``.

    """
    if target is Target.RVC4:
        from modelconverter.packages.rvc4.visualize import RVC4Visualizer

        return RVC4Visualizer(*args, **kwargs)
    raise ValueError(f"Visualizer not available for {target.name}")
