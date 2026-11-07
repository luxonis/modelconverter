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

from pathlib import Path

from typing_extensions import assert_never

from modelconverter.platforms.base_analyze import Analyzer
from modelconverter.platforms.base_benchmark import Benchmark
from modelconverter.platforms.base_exporter import Exporter
from modelconverter.platforms.base_inferer import Inferer
from modelconverter.platforms.base_visualize import Visualizer
from modelconverter.utils.config import SingleStageConfig
from modelconverter.utils.types import Platform


def get_exporter(
    platform: Platform, config: SingleStageConfig, output_dir: Path
) -> Exporter:
    """Create the `Exporter` implementation for the given platform.

    Args:
        platform: Platform the model is converted for.
        config: Configuration of the stage to convert.
        output_dir: Directory the converted model is written to.

    Returns:
        Exporter for the given platform.

    """
    if platform is Platform.RVC2:
        from modelconverter.platforms.rvc2.exporter import RVC2Exporter

        return RVC2Exporter(config, output_dir)

    if platform is Platform.RVC3:
        from modelconverter.platforms.rvc3.exporter import RVC3Exporter

        return RVC3Exporter(config, output_dir)

    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.exporter import RVC4Exporter

        return RVC4Exporter(config, output_dir)

    if platform is Platform.HAILO:  # pragma: no branch
        from modelconverter.platforms.hailo.exporter import HailoExporter

        return HailoExporter(config, output_dir)

    assert_never(platform)


def get_inferer(
    platform: Platform,
    model_path: str,
    src: Path,
    dest: Path,
    config: SingleStageConfig,
) -> Inferer:
    """Create the `Inferer` implementation for the given platform.

    The inferer is constructed using its ``from_config`` constructor.

    Args:
        platform: Platform the model was converted for.
        model_path: Path to the converted model.
        src: Directory holding the input files.
        dest: Directory the outputs are written to.
        config: Configuration of the stage to run.

    Returns:
        Inferer for the given platform.

    """
    if platform is Platform.RVC2:
        from modelconverter.platforms.rvc2.inferer import RVC2Inferer

        return RVC2Inferer.from_config(model_path, src, dest, config)

    if platform is Platform.RVC3:
        from modelconverter.platforms.rvc3.inferer import RVC3Inferer

        return RVC3Inferer.from_config(model_path, src, dest, config)

    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.inferer import RVC4Inferer

        return RVC4Inferer.from_config(model_path, src, dest, config)

    if platform is Platform.HAILO:  # pragma: no branch
        from modelconverter.platforms.hailo.inferer import HailoInferer

        return HailoInferer.from_config(model_path, src, dest, config)

    assert_never(platform)


def get_benchmark(platform: Platform, model_path: str) -> Benchmark:
    """Create the `Benchmark` implementation for the given platform.

    Args:
        platform: Platform to benchmark the model on.
        model_path: Model to benchmark.

    Returns:
        Benchmark for the given platform.

    Raises:
        NotImplementedError: If ``platform`` is ``Platform.HAILO``.

    """
    if platform is Platform.RVC2:
        from modelconverter.platforms.rvc2.benchmark import RVC2Benchmark

        return RVC2Benchmark(model_path)

    if platform is Platform.RVC3:
        from modelconverter.platforms.rvc3.benchmark import RVC3Benchmark

        return RVC3Benchmark(model_path)

    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.benchmark import RVC4Benchmark

        return RVC4Benchmark(model_path)

    if platform is Platform.HAILO:  # pragma: no branch
        raise NotImplementedError("Hailo Benchmark is not implemented yet.")

    assert_never(platform)


def get_analyzer(
    platform: Platform,
    device_ip: str | None,
    device_id: str | None,
    dlc_model_path: str,
    image_dirs: dict[str, str],
    *,
    image_subset: int | None = None,
) -> Analyzer:
    """Create the `Analyzer` implementation for the given platform.

    Only RVC4 provides an analyzer.

    Args:
        platform: Platform the model was converted for.
        device_ip: IP address of the device to analyze on.
        device_id: ID of the device to analyze on.
        dlc_model_path: Path or URL of the DLC model to analyze.
        image_dirs: Mapping of the model's input names to the
            directories holding the data for that input.
        image_subset: If given, only the first ``image_subset``
            supported files of every directory are used.

    Returns:
        Analyzer for the given platform.

    Raises:
        ValueError: If ``platform`` is not ``Platform.RVC4``.

    """
    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.analyze import RVC4Analyzer

        return RVC4Analyzer(
            device_ip,
            device_id,
            dlc_model_path,
            image_dirs,
            image_subset=image_subset,
        )

    raise ValueError(f"Analyzer not available for {platform.name}")


def get_visualizer(
    platform: Platform, dir_path: str | None = None
) -> Visualizer:
    """Create the `Visualizer` implementation for the given platform.

    Only RVC4 provides a visualizer.

    Args:
        platform: Platform the model was converted for.
        dir_path: Directory containing the analysis results. Defaults
            to the ``analysis`` subdirectory of the modelconverter
            output directory.

    Returns:
        Visualizer for the given platform.

    Raises:
        ValueError: If ``platform`` is not ``Platform.RVC4``.

    """
    if platform is Platform.RVC4:
        from modelconverter.platforms.rvc4.visualize import RVC4Visualizer

        return RVC4Visualizer(dir_path)
    raise ValueError(f"Visualizer not available for {platform.name}")
