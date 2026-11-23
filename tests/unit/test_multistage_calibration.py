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
import numpy.typing as npt
import pytest
from typing_extensions import override

from modelconverter.platforms import multistage_exporter
from modelconverter.platforms.base_inferer import Inferer
from modelconverter.platforms.multistage_exporter import MultiStageExporter
from modelconverter.platforms.rvc3.exporter import RVC3Exporter
from modelconverter.utils.config import (
    Config,
    ImageCalibrationConfig,
    LinkCalibrationConfig,
    OutputConfig,
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
        """Record the attributes ``MultiStageExporter`` reads.

        Args:
            config: Config of the stage being exported.
            output_dir: Directory the stage writes its outputs to.

        """
        self.config = config
        self.output_dir = output_dir
        self.inputs = {inp.name: inp for inp in config.inputs}
        self.model_name = config.input_model.stem
        self.inference_model_path = config.input_model


class StubInferer(Inferer):
    """A real ``Inferer`` with the vendor runtime replaced by ones."""

    @override
    def setup(self) -> None:
        """Do nothing -- there is no vendor runtime to set up."""

    @override
    def infer(
        self, inputs: dict[str, Path]
    ) -> dict[str, npt.NDArray[np.generic]]:
        """Return an array of ones for every output.

        Args:
            inputs: Mapping of input name to input file. Ignored, the
                outputs do not depend on them.

        Returns:
            Mapping of output name to an array of ones of that shape.

        """
        return {
            name: np.ones(shape, dtype=np.float32)
            for name, shape in self.out_shapes.items()
        }


@pytest.fixture
def calibration_dir(work_dir: Path) -> Path:
    """Create a directory of images, never decoded -- ``infer`` is stubbed."""
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
    were an output directory fails the whole conversion.
    """
    exporter._produce_calibration_data(exporter.exporters["second"])

    calibration = _resolved_calibration(config)
    assert not calibration.generated_from_link
    path = calibration.path
    produced = sorted(p.name for p in path.iterdir())
    assert produced == [f"{i}.npy" for i in range(N_CALIBRATION_IMAGES)]
    # (1, 10) and (1, 5, 5, 5) flattened and concatenated.
    assert np.load(path / "0.npy").shape == (135,)


def test_inference_results_are_marked(exporter: MultiStageExporter):
    """The marker is what lets a rerun clear its own results, so it has
    to be there even though the linking code skips it.
    """
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
    directly, so the marker never enters the picture.
    """
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


@pytest.mark.parametrize("platform", [Platform.RVC3, Platform.RVC4])
def test_direct_link_records_inferer_output_layout(
    exporter: MultiStageExporter, config: Config, platform: Platform
):
    exporter.platform = platform
    second = config.stages["second"]
    second.inputs[0].calibration = LinkCalibrationConfig(
        stage="first", output="output1"
    )

    exporter._produce_calibration_data(exporter.exporters["second"])

    calibration = _resolved_calibration(config)
    assert calibration.generated_from_link
    expected = (
        "NCHW"
        if platform is Platform.RVC4
        else config.stages["first"].outputs[1].layout
    )
    assert calibration.generated_layout == expected


@pytest.mark.parametrize("source_layout", ["NHWC", "NCDE"])
def test_direct_rvc3_link_uses_upstream_layout_for_feature_calibration(
    exporter: MultiStageExporter,
    config: Config,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_layout: str,
):
    """A feature map retains its channel order when stage layouts differ."""
    exporter.platform = Platform.RVC3
    # Include negative values and more than three channels, as a backbone's
    # feature map would. HWC is the sample order POT's NumPy reader expects.
    expected = np.arange(160, dtype=np.float32).reshape(4, 5, 8) - 80
    source = (
        expected[None]
        if source_layout == "NHWC"
        else expected.transpose(2, 0, 1)[None]
    )
    config.stages["first"].outputs = [
        OutputConfig(
            name="features", shape=list(source.shape), layout=source_layout
        )
    ]
    monkeypatch.setattr(
        StubInferer, "infer", lambda _self, _inputs: {"features": source}
    )
    inp = config.stages["second"].inputs[0]
    inp.shape = [1, 8, 4, 5]
    inp.layout = "NCHW"
    inp.calibration = LinkCalibrationConfig(stage="first", output="features")

    exporter._produce_calibration_data(exporter.exporters["second"])

    calibration = _resolved_calibration(config)
    pot = object.__new__(RVC3Exporter)
    pot.intermediate_outputs_dir = work_dir
    pot._write_calibration_tensors(
        inp, calibration, [calibration.path / "0.npy"], shape=inp.shape
    )
    np.testing.assert_array_equal(
        np.load(work_dir / "calibration_tensors/0.npy"), expected
    )


def test_direct_flat_output_is_not_mislabeled_as_rank_four(
    exporter: MultiStageExporter, config: Config
):
    """RVC4 keeps opaque buffers when a vector feeds a reshaped input."""
    inp = config.stages["second"].inputs[0]
    inp.shape = [1, 10, 1, 1]
    inp.layout = "NCHW"
    inp.calibration = LinkCalibrationConfig(stage="first", output="output0")

    exporter._produce_calibration_data(exporter.exporters["second"])

    calibration = _resolved_calibration(config)
    assert not calibration.generated_from_link
    assert np.load(calibration.path / "0.npy").shape == (1, 10)


@pytest.mark.parametrize("source_layout", ["CHW", "HWC"])
@pytest.mark.parametrize("input_layout", ["NCHW", "NCDE"])
def test_batchless_script_feature_map_uses_source_axes(
    exporter: MultiStageExporter,
    config: Config,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_layout: str,
    input_layout: str,
):
    """A script can drop the batch axis before feeding a generic feature input."""
    exporter.platform = Platform.RVC3
    expected = np.arange(160, dtype=np.float32).reshape(4, 5, 8) - 80
    source = (
        expected if source_layout == "HWC" else expected.transpose(2, 0, 1)
    )
    monkeypatch.setattr(
        StubInferer, "infer", lambda _self, _inputs: {"features": source[None]}
    )
    inp = config.stages["second"].inputs[0]
    inp.shape = [1, 8, 4, 5]
    inp.layout = input_layout
    inp.calibration = LinkCalibrationConfig(
        stage="first",
        script="def run_script(outputs):\n    return outputs['features'][0]\n",
        layout=source_layout,
    )

    exporter._produce_calibration_data(exporter.exporters["second"])

    calibration = _resolved_calibration(config)
    pot = object.__new__(RVC3Exporter)
    pot.intermediate_outputs_dir = work_dir
    pot._write_calibration_tensors(
        inp, calibration, [calibration.path / "0.npy"], shape=inp.shape
    )
    np.testing.assert_array_equal(
        np.load(work_dir / "calibration_tensors/0.npy"), expected
    )


@pytest.mark.parametrize("platform", list(Platform))
def test_script_link_records_declared_layout(
    exporter: MultiStageExporter, config: Config, platform: Platform
):
    exporter.platform = platform
    second = config.stages["second"]
    second.inputs[0].calibration = LinkCalibrationConfig(
        stage="first",
        script="def run_script(outputs):\n    return outputs['output1']\n",
        layout="NCHW",
    )

    exporter._produce_calibration_data(exporter.exporters["second"])

    calibration = _resolved_calibration(config)
    assert calibration.generated_from_link
    assert calibration.generated_layout == "NCHW"


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
