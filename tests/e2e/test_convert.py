"""Tier-2 "successfully convert" tests.

Each test fetches a representative Luxonis model-zoo NN archive (by slug,
via HubAI) and converts it for one backend using the real vendor tools.
It asserts the conversion succeeds and produces an output artifact; it
does NOT check numerical fidelity (deferred for now).

These run **inside the per-backend Docker image** (where the vendor tools
live) under ``pytest-cov``, calling the ``convert`` command in-process so
the exporter code is measured directly. Each backend case carries a
platform marker, so ``pytest -m rvc4`` runs only the RVC4 conversions.
"""

from pathlib import Path

import pytest

from modelconverter.__main__ import convert
from modelconverter.utils.constants import OUTPUTS_DIR
from modelconverter.utils.types import Target

from .models import (
    BACKENDS,
    ZooModel,
    download_zoo_archive,
    models_for_backend,
)

# Keep the slow Hailo compilation cheap for a smoke conversion.
_HAILO_FAST_OPTS = (
    "hailo.compression_level", "0",
    "hailo.optimization_level", "0",
    "hailo.disable_compilation", "True",
)

_CASES = [
    pytest.param(
        backend,
        model,
        marks=getattr(pytest.mark, backend),
        id=f"{backend}-{model.slug}",
    )
    for backend in BACKENDS
    for model in models_for_backend(backend)
]


def _assert_produced(output_name: str) -> None:
    out_dir = OUTPUTS_DIR / output_name
    assert out_dir.exists(), f"output dir {out_dir} was not created"
    produced = list(out_dir.rglob("*.tar.xz")) + list(out_dir.rglob("*.tar"))
    assert produced, f"no NN archive produced in {out_dir}"


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize(("backend", "model"), _CASES)
def test_convert_archive(backend: str, model: ZooModel, tmp_path: Path):
    archive = download_zoo_archive(model, backend, tmp_path / "archive")
    output_name = f"_{model.slug}-{backend}-e2e"
    extra = _HAILO_FAST_OPTS if backend == "hailo" else ()
    convert(
        Target(backend),
        "calibration.max_images",
        "30",
        *extra,
        path=str(archive),
        output_dir=output_name,
        to="nn_archive",
    )
    _assert_produced(output_name)


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.rvc4
@pytest.mark.parametrize("to_format", ["native", "nn_archive"])
def test_convert_output_formats(to_format: str, tmp_path: Path):
    """Exercises both the native and nn_archive output branches of the
    ``convert`` command on RVC4."""
    model = models_for_backend("rvc4")[0]
    archive = download_zoo_archive(model, "rvc4", tmp_path / "archive")
    output_name = f"_{model.slug}-rvc4-{to_format}-e2e"
    convert(
        Target.RVC4,
        "calibration.max_images",
        "30",
        path=str(archive),
        output_dir=output_name,
        to=to_format,
    )
    if to_format == "nn_archive":
        _assert_produced(output_name)
    else:
        assert (OUTPUTS_DIR / output_name).exists()
