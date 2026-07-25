"""Representative Luxonis model-zoo models for the Tier-2 "successfully
convert" e2e tests.

Per the project decision, the NN archives are fetched *fresh from the
model zoo* (HubAI) by slug at test time. Each entry is a small,
representative model covering one of the main task families so the
conversion path is exercised end-to-end for every backend.

Conversion consumes the model's **ONNX** instance (a ``.tar.xz`` NN
archive holding the ONNX + ``config.json``); the per-platform (RVC2/RVC4/…)
instances are the *outputs* of conversion and are ignored here. Download
uses the official ``hubai-sdk``: resolve the model by its (team-less)
slug, list its public instances, pick the ONNX one, and
``download_instance`` it. Requires ``HUBAI_API_KEY`` + network; when the
archive cannot be fetched the calling test is skipped (not failed) -- the
fetch is a fixture concern, not the thing under test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class ZooModel:
    slug: str
    """Short identifier used in the test id / output dir."""
    model_slug: str
    """HubAI model slug (no team prefix), e.g. ``yolov6-nano``."""
    onnx_variant: str
    """Preferred ONNX instance slug to disambiguate (e.g. resolution)."""
    task: str
    """Task family (detection / segmentation / …)."""
    multistage: bool = False


# A few representative model-zoo models (slugs shared with the benchmark
# suite / verified against HubAI). Keep the list short -- the e2e tier is a
# smoke "does it convert", not breadth.
MODELS: list[ZooModel] = [
    ZooModel(
        slug="yolov6-nano",
        model_slug="yolov6-nano",
        onnx_variant="r2-coco-512x288",
        task="detection",
    ),
    ZooModel(
        slug="yolov8-instance-seg-nano",
        model_slug="yolov8-instance-segmentation-nano",
        onnx_variant="coco-512x288",
        task="segmentation",
        multistage=True,
    ),
    ZooModel(
        slug="deeplab-v3-plus",
        model_slug="deeplab-v3-plus",
        onnx_variant="512x288",
        task="segmentation",
    ),
]

# Backends exercised by the e2e matrix. Hailo runs on a larger CI runner.
BACKENDS: list[str] = ["rvc2", "rvc3", "rvc4", "hailo"]

# A trimmed set for the slower Hailo backend (keep CI time reasonable).
_SLOW_BACKENDS = {"hailo"}
_SLOW_MODEL_SLUGS = {"yolov6-nano"}


def models_for_backend(backend: str) -> list[ZooModel]:
    """The model subset to convert for a given backend."""
    if backend in _SLOW_BACKENDS:
        return [m for m in MODELS if m.slug in _SLOW_MODEL_SLUGS]
    return list(MODELS)


def download_zoo_archive(model: ZooModel, backend: str, dest: Path) -> Path:
    """Downloads ``model``'s ONNX NN archive from the zoo into ``dest``.

    Skips the calling test (via ``pytest.skip``) when the archive cannot
    be fetched -- e.g. no ``HUBAI_API_KEY``, no network, or the instance is
    not available.
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from hubai_sdk import HubAIClient

        client = HubAIClient(api_key=os.environ["HUBAI_API_KEY"])
        hub_model = client.models.get_model(model.model_slug)
        instances = client.instances.list_instances(
            model_id=hub_model.id, is_public=True
        )
        onnx = [
            i
            for i in instances
            if str(getattr(i, "model_type", "")).endswith("ONNX")
        ]
        if not onnx:
            raise RuntimeError(f"no ONNX instance for {model.model_slug!r}")
        chosen = next(
            (i for i in onnx if i.slug == model.onnx_variant), onnx[0]
        )
        return Path(
            client.instances.download_instance(
                chosen.id, output_dir=str(dest)
            )
        )
    except Exception as exc:
        pytest.skip(
            f"Could not fetch zoo archive for {model.model_slug!r} "
            f"({backend}): {exc}"
        )
