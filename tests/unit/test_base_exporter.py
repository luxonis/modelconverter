"""Host-side tests for what ``Exporter`` lays out beside the model.

The conversion itself needs the vendor tools, but the constructor's
bookkeeping -- the copies it makes into the output and intermediate
directories -- is plain filesystem work and is what a user is left with
afterwards.
"""

import json
from pathlib import Path

import pytest

from modelconverter.platforms.base_exporter import Exporter
from modelconverter.utils.config import Config, SingleStageConfig
from modelconverter.utils.types import Platform
from tests.helpers.onnx_factory import multi_file_external_onnx


class StubExporter(Exporter):
    """``Exporter`` with only its abstract vendor half stubbed out."""

    platform = Platform.RVC2

    def exporter_buildinfo(self) -> dict:
        """Return empty build info."""
        return {}

    def export(self) -> Path:
        """Return the input model unchanged."""
        return self._input_model


@pytest.fixture
def stage_config(work_dir: Path) -> tuple[SingleStageConfig, list[Path]]:
    model = work_dir / "model.onnx"
    external_data = multi_file_external_onnx(model)
    assert len(external_data) > 1
    config = Config.get_config(
        None,
        {
            "input_model": str(model),
            "shape": [1, 3, 8, 8],
            "onnx_simplification": False,
        },
    )
    return next(iter(config.stages.values())), external_data


def test_every_external_data_file_travels_with_the_model(
    work_dir: Path, stage_config: tuple[SingleStageConfig, list[Path]]
):
    """A model saved with ``all_tensors_to_one_file=False`` keeps one
    companion file per tensor.

    Copying only the first leaves the ``.onnx`` in the output directory
    referencing weights that are not there, so the user cannot load what
    the conversion handed them.
    """
    config, external_data = stage_config

    exporter = StubExporter(config=config, output_dir=work_dir / "output")

    for data in external_data:
        assert (exporter.output_dir / data.name).read_bytes() == (
            data.read_bytes()
        )
        assert (
            exporter.intermediate_outputs_dir / data.name
        ).read_bytes() == data.read_bytes()


def test_run_moves_the_export_and_records_the_buildinfo(
    work_dir: Path, stage_config: tuple[SingleStageConfig, list[Path]]
):
    """`run` is the half of the exporter that is not vendor-specific.

    It calls both abstract methods, renames what `export` produced to
    the original model name, and leaves the buildinfo beside it.
    """
    config, _ = stage_config

    exporter = StubExporter(config=config, output_dir=work_dir / "output")
    output = exporter.run()

    assert output.is_file()
    assert output.parent == exporter.output_dir
    buildinfo = exporter.output_dir / "buildinfo.json"
    assert buildinfo.is_file()
    assert "modelconverter_version" in json.loads(buildinfo.read_text())
