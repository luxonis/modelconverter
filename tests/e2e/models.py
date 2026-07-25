"""Models / assets for the Tier-2 "successfully convert" e2e tests.

Two sources are used:

* **Zoo archives** (fresh from HubAI, via ``hubai-sdk``) for single-stage
  smoke conversions -- their ONNX instance is a ``.tar.xz`` NN archive with
  an image input, so the default (random) calibration converts cleanly on
  every backend.
* **Purpose-built test-bucket archives/configs** for the multistage +
  cross-format scenarios. A raw zoo multistage archive cannot be converted
  with calibration (its postprocessor stage's non-image inputs get random
  calibration, which fails), so the multistage case uses the
  ``yolov8n_seg`` archive/config that carries the full **linked**
  calibration (stage-2 inputs calibrated from stage-1 outputs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

# Purpose-built assets carrying their own (full/linked) calibration.
GS_PREFIX = "gs://luxonis-test-bucket/modelconverter"

# Backends exercised by the e2e matrix. Hailo runs on a larger CI runner.
BACKENDS: list[str] = ["rvc2", "rvc3", "rvc4", "hailo"]


@dataclass(frozen=True)
class ZooModel:
    slug: str
    """Short identifier used in the test id / output dir."""
    model_slug: str
    """HubAI model slug (no team prefix), e.g. ``yolov6-nano``."""
    onnx_variant: str
    """Preferred ONNX instance slug to disambiguate (e.g. resolution)."""


# Single-stage model(s) used for the per-backend smoke conversion.
SINGLE_STAGE_MODELS: list[ZooModel] = [
    ZooModel(
        slug="yolov6-nano",
        model_slug="yolov6-nano",
        onnx_variant="r2-coco-512x288",
    ),
]


def single_stage_models_for_backend(backend: str) -> list[ZooModel]:
    """The single-stage model subset to convert for a given backend."""
    return list(SINGLE_STAGE_MODELS)


def download_zoo_archive(model: ZooModel, dest: Path) -> Path:
    """Downloads ``model``'s ONNX NN archive fresh from the zoo into ``dest``.

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
        pytest.skip(f"Could not fetch zoo archive for {model.model_slug!r}: {exc}")
