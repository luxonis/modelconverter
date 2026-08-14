"""Host-side tests for the multistage exporter's linked calibration.

A stage whose calibration links to another stage gets its data by running
that stage's inferer. Exporters live in vendor-heavy modules, so
``get_exporter`` is stubbed out; the inferer keeps the real ``Inferer``
base over a fake forward pass, because its bookkeeping -- the per-output
directories and the results marker it leaves next to them -- is exactly
what the linking code has to read back.
"""

from pathlib import Path

import numpy as np
import pytest

from modelconverter.platforms import multistage_exporter
from modelconverter.platforms.base_inferer import Inferer
from modelconverter.platforms.multistage_exporter import MultiStageExporter
from modelconverter.utils.config import (
    Config,
    ImageCalibrationConfig,
    LinkCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.constants import INFERENCE_MARKER
from modelconverter.utils.types import Platform

# Concatenates every output of the linked stage into one calibration
# tensor. `standard_dummy_onnx` has a (1, 10) and a (1, 5, 5, 5) output.
LINK_SCRIPT = """
import numpy as np

def run_script(outputs):
    return np.concatenate([outputs[name].flatten() for name in sorted(outputs)])
"""

# Takes only the first output, so its results are distinguishable from
# `LINK_SCRIPT`'s by shape alone.
FIRST_OUTPUT_SCRIPT = """
def run_script(outputs):
    return outputs[sorted(outputs)[0]].flatten()
"""

N_CALIBRATION_IMAGES = 3


class StubExporter:
    """The slice of ``Exporter`` that ``MultiStageExporter`` uses."""

    def __init__(self, config: SingleStageConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.inputs = {inp.name: inp for inp in config.inputs}
        self.model_name = config.input_model.stem
        self.inference_model_path = config.input_model


class StubInferer(Inferer):
    """A real ``Inferer`` with the vendor runtime replaced by ones."""

    def setup(self) -> None: ...

    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]:
        return {
            name: np.ones(shape, dtype=np.float32)
            for name, shape in self.out_shapes.items()
        }


@pytest.fixture
def calibration_dir(work_dir: Path) -> Path:
    """A directory of images, never decoded -- ``infer`` is stubbed."""
    path = work_dir / "calibration"
    path.mkdir()
    for i in range(N_CALIBRATION_IMAGES):
        (path / f"{i}.png").write_bytes(b"")
    return path


@pytest.fixture
def config(dummy_onnx: Path, calibration_dir: Path) -> Config:
    """Two stages, the second calibrated from the first via a script."""
    calibration = {
        "path": str(calibration_dir),
        "max_images": N_CALIBRATION_IMAGES,
    }
    return Config.get_config(
        None,
        {
            "name": "pipeline",
            "stages": {
                "first": {
                    "input_model": str(dummy_onnx),
                    "inputs": [
                        {"name": "input0", "calibration": calibration},
                        {"name": "input1", "calibration": calibration},
                    ],
                },
                "second": {
                    "input_model": str(dummy_onnx),
                    "inputs": [
                        {
                            "name": "input0",
                            "calibration": {
                                "stage": "first",
                                "script": LINK_SCRIPT,
                            },
                        },
                        {"name": "input1", "calibration": calibration},
                    ],
                },
            },
        },
    )


@pytest.fixture
def exporter(
    config: Config, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> MultiStageExporter:
    monkeypatch.setattr(
        multistage_exporter,
        "get_exporter",
        lambda _platform, stage_config, output_dir: StubExporter(
            stage_config, output_dir
        ),
    )
    monkeypatch.setattr(
        multistage_exporter,
        "get_inferer",
        lambda _platform, model_path, src, dest, stage_config: (
            StubInferer.from_config(model_path, src, dest, stage_config)
        ),
    )
    return MultiStageExporter(Platform.RVC4, config, work_dir / "output")


def _resolved_calibration(config: Config) -> ImageCalibrationConfig:
    calibration = config.stages["second"].inputs[0].calibration
    assert isinstance(calibration, ImageCalibrationConfig)
    return calibration


def test_script_calibration_uses_only_the_output_dirs(
    exporter: MultiStageExporter, config: Config
):
    """The inferer marks its destination with a file that sits next to
    the per-output directories; feeding that file to the script as if it
    were an output directory fails the whole conversion."""
    exporter._produce_calibration_data(exporter.exporters["second"])

    path = _resolved_calibration(config).path
    produced = sorted(p.name for p in path.iterdir())
    assert produced == [f"{i}.npy" for i in range(N_CALIBRATION_IMAGES)]
    # (1, 10) and (1, 5, 5, 5) flattened and concatenated.
    assert np.load(path / "0.npy").shape == (135,)


def test_inference_results_are_marked(exporter: MultiStageExporter):
    """The marker is what lets a rerun clear its own results, so it has
    to be there even though the linking code skips it."""
    exporter._produce_calibration_data(exporter.exporters["second"])

    results_dir = (
        exporter._intermediate_outputs_dir / "dummy_model_calibration"
    )
    assert (results_dir / INFERENCE_MARKER).is_file()
    assert sorted(p.name for p in results_dir.iterdir() if p.is_dir()) == [
        "output0",
        "output1",
    ]


def test_linked_output_calibration_points_at_the_output_dir(
    exporter: MultiStageExporter, config: Config
):
    """The ``output`` form of a link names one output directory
    directly, so the marker never enters the picture."""
    second = config.stages["second"]
    second.inputs[0].calibration = LinkCalibrationConfig(
        stage="first", output="output0"
    )

    exporter._produce_calibration_data(exporter.exporters["second"])

    path = _resolved_calibration(config).path
    assert path.name == "output0"
    assert sorted(p.name for p in path.iterdir()) == [
        f"{i}.npy" for i in range(N_CALIBRATION_IMAGES)
    ]


def test_two_script_linked_inputs_calibrate_with_their_own_scripts(
    exporter: MultiStageExporter, config: Config
):
    """Both inputs of a stage may be calibrated from the same previous
    stage, each with a script of its own.

    The generated data is keyed by the receiving input, not just the
    linked stage; sharing one directory would leave both inputs
    calibrated on whichever script ran last.
    """
    second = config.stages["second"]
    second.inputs[1].calibration = LinkCalibrationConfig(
        stage="first", script=FIRST_OUTPUT_SCRIPT
    )

    exporter._produce_calibration_data(exporter.exporters["second"])

    shapes = []
    for inp in second.inputs:
        assert isinstance(inp.calibration, ImageCalibrationConfig)
        assert (inp.calibration.path.parent / "script.py").is_file()
        assert sorted(p.name for p in inp.calibration.path.iterdir()) == [
            f"{i}.npy" for i in range(N_CALIBRATION_IMAGES)
        ]
        shapes.append(np.load(inp.calibration.path / "0.npy").shape)
    # (1, 10) and (1, 5, 5, 5) concatenated for one input; only (1, 10)
    # flattened for the other.
    assert shapes == [(135,), (10,)]
