"""Representative Luxonis model-zoo models for the Tier-2 "successfully
convert" e2e tests.

Per the project decision, the NN archives are fetched *fresh from the
model zoo* (HubAI) by slug at test time rather than from a pre-staged
bucket. Each entry is a small, representative model covering one of the
main task families (classification / detection / instance-segmentation)
so the conversion path is exercised end-to-end for every backend.

The download uses the in-repo HubAI :class:`Request` API to resolve a
model-instance slug to an id and fetch its NN archive. It requires
``HUBAI_API_KEY`` and network access; when either is missing (or the
model instance cannot be resolved) the e2e test that needs it is skipped
rather than failed -- the archive fetch is a fixture concern, not the
thing under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from modelconverter.utils.filesystem_utils import download_from_remote


@dataclass(frozen=True)
class ZooModel:
    slug: str
    """Short identifier used in the test id."""
    hub_slug: str
    """Full HubAI model slug (``<team>/<model>``)."""
    task: str
    """Task family (classification / detection / segmentation)."""
    variants: dict[str, str] = field(default_factory=dict)
    """Optional per-backend ``modelInstances`` slug overrides."""
    multistage: bool = False


# A few representative model-zoo models. Keep this list short -- the point
# of the e2e tier is a smoke "does it convert", not breadth.
MODELS: list[ZooModel] = [
    ZooModel(
        slug="efficientnet-lite",
        hub_slug="luxonis/efficientnet-lite",
        task="classification",
    ),
    ZooModel(
        slug="yolov6-nano",
        hub_slug="luxonis/yolov6-nano",
        task="detection",
    ),
    ZooModel(
        slug="yolov8-instance-segmentation-nano",
        hub_slug="luxonis/yolov8-instance-segmentation-nano",
        task="segmentation",
        multistage=True,
    ),
]

# Backends exercised by the e2e matrix. Hailo runs on a larger CI runner.
BACKENDS: list[str] = ["rvc2", "rvc3", "rvc4", "hailo"]

# A trimmed set for the slower Hailo backend (keep CI time reasonable).
_SLOW_BACKENDS = {"hailo"}
_SLOW_MODEL_SLUGS = {"efficientnet-lite"}


def models_for_backend(backend: str) -> list[ZooModel]:
    """The model subset to convert for a given backend."""
    if backend in _SLOW_BACKENDS:
        return [m for m in MODELS if m.slug in _SLOW_MODEL_SLUGS]
    return list(MODELS)


def _resolve_instance_download_url(model: ZooModel, backend: str) -> str:
    """Resolves a model to its HubAI NN-archive download URL.

    Uses the in-repo HubAI API client. Raises on any failure so callers
    can convert it into a ``pytest.skip``.
    """
    from modelconverter.cli import slug_to_id
    from modelconverter.utils.hub_requests import Request

    instance_slug = model.variants.get(backend, model.hub_slug)
    instance_id = slug_to_id(instance_slug, "modelInstances")
    # The download endpoint returns the archive's signed URL(s).
    data = Request.get(f"modelInstances/{instance_id}/download")
    if isinstance(data, list):
        urls = [d.get("download_url") or d.get("url") for d in data]
        url = next((u for u in urls if u and u.endswith((".tar.xz", ".tar"))), None)
    else:
        url = data.get("download_url") or data.get("url")
    if not url:
        raise RuntimeError(
            f"No NN-archive download URL for {instance_slug!r}"
        )
    return url


def download_zoo_archive(model: ZooModel, backend: str, dest: Path) -> Path:
    """Downloads ``model``'s NN archive from the zoo into ``dest``.

    Skips the calling test (via ``pytest.skip``) when the archive cannot
    be fetched -- e.g. no ``HUBAI_API_KEY``, no network, or the instance
    is not published for this backend.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        url = _resolve_instance_download_url(model, backend)
        return download_from_remote(url, dest)
    except Exception as exc:
        pytest.skip(
            f"Could not fetch zoo archive for {model.hub_slug!r} "
            f"({backend}): {exc}"
        )
