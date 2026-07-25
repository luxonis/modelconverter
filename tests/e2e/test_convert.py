"""Tier-2 "successfully convert" tests.

These run inside the per-backend Docker image (`modelconverter shell
<backend> --dev -c pytest ...`) where the vendor tools live; the tests
call ``convert`` in-process (``shell`` sets IN_DOCKER, so the launcher is
skipped) so pytest-cov measures the exporter code directly. They assert a
conversion succeeds and produces an output artifact -- not numerical
fidelity.

Scenarios:

* per-backend single-stage smoke, from a fresh zoo NN archive;
* an rvc4 cross-format matrix over single-stage (``resnet18``) and
  **multistage (``yolov8n_seg``, full linked calibration)** models and
  every ``{nn_archive, native}`` input/output combination.

Each case carries its backend marker, so ``pytest -m rvc4`` runs only the
RVC4 conversions.
"""

from pathlib import Path

import pytest

from modelconverter.__main__ import convert
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target

from .models import (
    BACKENDS,
    GS_PREFIX,
    ZooModel,
    download_zoo_archive,
    single_stage_models_for_backend,
)


def _assert_archive_produced(output_name: str) -> None:
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    produced = list(out_dir.rglob("*.tar.xz")) + list(out_dir.rglob("*.tar"))
    assert produced, f"no NN archive produced in {out_dir}"


def _assert_output_produced(output_name: str) -> None:
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    assert any(out_dir.iterdir()), f"no output produced in {out_dir}"


# --------------------------------------------------------------------------- #
# Per-backend single-stage smoke (fresh zoo archive)                          #
# --------------------------------------------------------------------------- #

_SINGLE_STAGE_CASES = [
    pytest.param(
        backend,
        model,
        marks=getattr(pytest.mark, backend),
        id=f"{backend}-{model.slug}",
    )
    for backend in BACKENDS
    for model in single_stage_models_for_backend(backend)
]


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize(("backend", "model"), _SINGLE_STAGE_CASES)
def test_convert_single_stage(backend: str, model: ZooModel, tmp_path: Path):
    archive = download_zoo_archive(model, tmp_path / "archive")
    output_name = f"_{model.slug}-{backend}-e2e"
    # Keep the (otherwise very slow) Hailo compilation cheap for a smoke run.
    extra = (
        (
            "hailo.compression_level",
            "0",
            "hailo.optimization_level",
            "0",
            "hailo.disable_compilation",
            "True",
        )
        if backend == "hailo"
        else ()
    )
    convert(
        Target(backend),
        *extra,
        path=str(archive),
        output_dir=output_name,
        to="nn_archive",
    )
    _assert_archive_produced(output_name)


# --------------------------------------------------------------------------- #
# RVC4 cross-format matrix: single-stage + multistage, all I/O formats        #
# --------------------------------------------------------------------------- #

# ``yolov8n_seg`` is multistage and carries full linked calibration.
# native -> nn_archive is unsupported for the multistage model, so it is
# excluded (matches the historical, passing matrix).
_CROSS_FORMAT_CASES = [
    (from_format, to_format, model)
    for from_format in ("nn_archive", "native")
    for to_format in ("nn_archive", "native")
    for model in ("resnet18", "yolov8n_seg")
    if (from_format, to_format, model)
    != ("native", "nn_archive", "yolov8n_seg")
]


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.rvc4
@pytest.mark.parametrize(
    ("from_format", "to_format", "model"), _CROSS_FORMAT_CASES
)
def test_rvc4_cross_format(from_format: str, to_format: str, model: str):
    # nn_archive input -> .tar.xz; native input -> the .yaml config (both
    # carry their own calibration, incl. the multistage linked calibration).
    suffix = "tar.xz" if from_format == "nn_archive" else "yaml"
    url = f"{GS_PREFIX}/{model}.{suffix}"
    output_name = f"_{model}-{from_format}-to-{to_format}-e2e"
    convert(Target.RVC4, path=url, output_dir=output_name, to=to_format)
    if to_format == "nn_archive":
        _assert_archive_produced(output_name)
    else:
        _assert_output_produced(output_name)
