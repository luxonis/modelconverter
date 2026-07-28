"""Task-metric regression tests for real public HubAI source models.

Unlike `test_precision`, these tests score predictions against labelled
images.  They use Luxonis Eval only as a test library: ModelConverter still
does conversion and runs the resulting native artifact through its own
inferers, so no Luxonis device is needed.

Run in an RVC2/RVC4 development image::

    modelconverter shell rvc2 --dev -c \
      'python -m pytest tests/conversion/test_model_evaluation.py -m rvc2'
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.packages.getters import get_inferer
from modelconverter.utils.config import SingleStageConfig
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    OUTPUTS_DIR,
)
from modelconverter.utils.general import sanitize_net_name
from modelconverter.utils.types import Target
from tests.helpers.evaluation import assert_quality, ordered_outputs
from tests.helpers.onnx_reference import ONNXReferenceInferer
from tests.helpers.precision import locate_converted_model
from tests.helpers.target_options import target_options

COCO_SAMPLE_FIXTURE = (
    "gs://luxonis-test-bucket/luxonis-ml-test-data/coco_sample.zip"
)
DATASET_NAME = "modelconverter-coco-sample-evaluation"
CALIBRATION_SPEC = f"{DATASET_NAME}:train"
YOLOV8_COEFFS_CALIBRATION_SCRIPT = """
def run_script(outputs):
    import numpy as np

    best_score = -np.inf
    best_coeffs = None
    for index in range(1, 4):
        scores = outputs[f"output{index}_yolov8"][0, 4:]
        flat_index = int(np.argmax(scores))
        score = float(scores.flat[flat_index])
        if score > best_score:
            _, y, x = np.unravel_index(flat_index, scores.shape)
            best_score = score
            best_coeffs = outputs[f"output{index}_masks"][0, :, y, x]
    return best_coeffs[None, :]
"""


@dataclass(frozen=True)
class QualityBudget:
    """Fixed quality gate derived from the initial RVC2/RVC4 baselines."""

    source_floors: dict[str, float]
    max_drops: dict[str, dict[str, float]]


@dataclass(frozen=True)
class EvaluationCase:
    """A pinned public source instance and its Luxonis Eval contract."""

    id: str
    instance_id: str
    parser_name: str
    parser_params: dict[str, Any]
    metric_names: tuple[str, ...]
    budget: QualityBudget


# The source instances are immutable HubAI IDs, not mutable model slugs. The
# floors reject broken source artifacts; converted models may lose only a small
# absolute amount of COCO-sample AP versus that source output.
CASES = (
    EvaluationCase(
        id="yolov6n-detection",
        instance_id="aimi_LCEFX2rJSsMhEjsyeMEcWn",
        parser_name="detection",
        parser_params={"subtype": "yolov6r2", "n_classes": 80},
        metric_names=("bbox",),
        budget=QualityBudget(
            source_floors={"bbox.AP": 0.14, "bbox.AP50": 0.22},
            max_drops={
                "rvc2": {"bbox.AP": 0.03, "bbox.AP50": 0.03},
                "rvc4": {"bbox.AP": 0.04, "bbox.AP50": 0.04},
            },
        ),
    ),
    EvaluationCase(
        id="yolov8n-instance-segmentation",
        instance_id="aimi_QtcY6CKM2cxB2QpmHVU4QL",
        parser_name="instance_segmentation",
        parser_params={"subtype": "yolov8", "n_classes": 80},
        metric_names=("bbox", "mask"),
        budget=QualityBudget(
            source_floors={
                "bbox.AP": 0.15,
                "bbox.AP50": 0.25,
                "mask.AP": 0.04,
                "mask.AP50": 0.135,
            },
            max_drops={
                "rvc2": {
                    "bbox.AP": 0.03,
                    "bbox.AP50": 0.03,
                    "mask.AP": 0.03,
                    "mask.AP50": 0.03,
                },
                "rvc4": {
                    "bbox.AP": 0.04,
                    "bbox.AP50": 0.04,
                    "mask.AP": 0.04,
                    "mask.AP50": 0.04,
                },
            },
        ),
    ),
)


def _lazy_eval_components(
    case: EvaluationCase,
) -> tuple[
    Any,
    list[tuple[str, Callable[[], Any]]],
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
]:
    """Load Luxonis Eval only in the selected RVC2/RVC4 test cases."""
    pytest.importorskip(
        "luxonis_eval",
        reason="install requirements-eval.txt to run real-model metric tests",
    )
    from luxonis_eval.metrics import (
        BboxMeanAveragePrecision,
        MaskMeanAveragePrecision,
    )
    from luxonis_eval.parsers import (
        YOLODetectionParser,
        YOLOInstanceSegmentationParser,
    )
    from luxonis_eval.utils.utils import (
        get_class_index_mapping,
        get_dataset_class_mapping,
        get_metric_ctx,
    )

    parser = (
        YOLODetectionParser()
        if case.parser_name == "detection"
        else YOLOInstanceSegmentationParser()
    )
    metrics: list[tuple[str, Callable[[], Any]]] = []
    if "bbox" in case.metric_names:
        metrics.append(("bbox", BboxMeanAveragePrecision))
    if "mask" in case.metric_names:
        metrics.append(("mask", MaskMeanAveragePrecision))
    return (
        parser,
        metrics,
        get_class_index_mapping,
        get_dataset_class_mapping,
        get_metric_ctx,
    )


def _image_shape(stage: SingleStageConfig) -> tuple[int, int]:
    """Return the configured NCHW input height and width."""
    assert len(stage.inputs) == 1
    input_cfg = stage.inputs[0]
    assert input_cfg.shape is not None
    assert input_cfg.layout == "NCHW"
    return input_cfg.shape[2], input_cfg.shape[3]


@pytest.fixture(scope="module")
def fixture_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> Any:
    """Parse the shared COCO sample GCS archive into an LDF."""
    from luxonis_ml.data import LuxonisLoader, LuxonisParser
    from luxonis_ml.utils.environ import environ

    work_dir = tmp_path_factory.mktemp("modelconverter-eval-fixture")
    old_base_path = environ.LUXONISML_BASE_PATH
    environ.LUXONISML_BASE_PATH = work_dir / "luxonis_ml"
    parser = LuxonisParser(
        COCO_SAMPLE_FIXTURE,
        dataset_name=DATASET_NAME,
        delete_local=True,
        save_dir=work_dir / "parsed",
    )
    dataset = parser.parse()
    loader = LuxonisLoader(
        dataset,
        view=["train", "val", "test"],
        height=288,
        width=512,
        keep_aspect_ratio=False,
        color_space="BGR",
    )
    try:
        yield loader
    finally:
        dataset.delete_dataset(delete_local=True)
        shutil.rmtree(CALIBRATION_DIR / DATASET_NAME, ignore_errors=True)
        environ.LUXONISML_BASE_PATH = old_base_path


def _download_source_model(instance_id: str, work_dir: Path) -> Path:
    """Download one immutable public HubAI source archive into ``work_dir``."""
    from hubai_sdk import HubAIClient

    return HubAIClient().instances.download_instance(
        instance_id, str(work_dir)
    )


def _calibration_images() -> list[Path]:
    """Return the images materialized by ModelConverter's LDF loader."""
    image_directory = CALIBRATION_DIR / DATASET_NAME
    paths = sorted(
        image_directory.glob("*.png"), key=lambda path: int(path.stem)
    )
    assert paths
    return paths


def _metrics_by_name(
    metrics: list[tuple[str, Callable[[], Any]]],
) -> dict[str, Any]:
    return {name: factory() for name, factory in metrics}


def _metric_results(metrics: dict[str, Any]) -> dict[str, float]:
    results: dict[str, float] = {}
    for name, metric in metrics.items():
        for key, value in metric.compute().items():
            if key != "metric":
                results[f"{name}.{key}"] = float(value)
    return results


_PARAMS = [
    pytest.param(
        platform,
        case,
        marks=getattr(pytest.mark, platform),
        id=f"{platform}-{case.id}",
    )
    for platform in ("rvc2", "rvc4")
    for case in CASES
]


@pytest.mark.real_model
@pytest.mark.parametrize(("platform", "case"), _PARAMS)
def test_real_model_task_metrics(
    platform: str,
    case: EvaluationCase,
    tmp_path: Path,
    fixture_dataset: Any,
) -> None:
    """Converted native outputs retain source quality on labelled COCO data."""
    (
        parser,
        metric_factories,
        get_class_index_mapping,
        get_dataset_class_mapping,
        get_metric_ctx,
    ) = _lazy_eval_components(case)
    target = Target(platform)
    options = target_options(target)
    source_archive = _download_source_model(
        case.instance_id, tmp_path / "models"
    )

    cfg, _, _ = get_configs(target, str(source_archive), list(options))
    stage_name, stage = next(iter(cfg.stages.items()))
    height, width = _image_shape(stage)
    assert (height, width) == (288, 512)

    loader = fixture_dataset

    output_name = sanitize_net_name(f"eval_{platform}_{case.id}")
    calibration_options = [
        *options,
        f"stages.{stage_name}.calibration.path",
        CALIBRATION_SPEC,
        f"stages.{stage_name}.calibration.max_images",
        "16",
    ]
    if case.parser_name == "instance_segmentation":
        post_stage_name, post_stage = next(
            (name, config)
            for name, config in cfg.stages.items()
            if name != stage_name
        )
        post_input_indices = {
            config.name: index
            for index, config in enumerate(post_stage.inputs)
        }
        assert "prototypes" in post_input_indices
        assert "coeffs" in post_input_indices

        prototypes_index = post_input_indices["prototypes"]
        coeffs_index = post_input_indices["coeffs"]

        calibration_options.extend(
            [
                f"stages.{post_stage_name}.inputs.{prototypes_index}.name",
                "prototypes",
                f"stages.{post_stage_name}.inputs.{prototypes_index}.calibration.stage",
                stage_name,
                f"stages.{post_stage_name}.inputs.{prototypes_index}.calibration.output",
                "protos_output",
                f"stages.{post_stage_name}.inputs.{coeffs_index}.name",
                "coeffs",
                f"stages.{post_stage_name}.inputs.{coeffs_index}.calibration.stage",
                stage_name,
                f"stages.{post_stage_name}.inputs.{coeffs_index}.calibration.script",
                YOLOV8_COEFFS_CALIBRATION_SCRIPT,
            ]
        )
    try:
        convert(
            target,
            *calibration_options,
            path=str(source_archive),
            output_dir=output_name,
            to="native",
        )
        calibration_images = _calibration_images()
        assert len(calibration_images) == len(loader)

        model_path = locate_converted_model(
            OUTPUTS_DIR / output_name, platform
        )
        reference = ONNXReferenceInferer.from_stage(stage)
        inferer = get_inferer(
            target,
            str(model_path),
            calibration_images[0].parent,
            OUTPUTS_DIR / f"{output_name}_infer",
            stage,
        )
        output_names = [output.name for output in stage.outputs]
        ldf_class_map = {
            class_index: class_name
            for class_name, class_index in loader._classes[""].items()
        }
        class_map = get_dataset_class_mapping("coco")
        class_index_map = get_class_index_mapping(ldf_class_map, class_map)
        metric_ctx = get_metric_ctx(
            {},
            width=width,
            height=height,
            ldf_class_map=ldf_class_map,
            class_map=class_map,
            class_index_map=class_index_map,
        )
        source_metrics = _metrics_by_name(metric_factories)
        converted_metrics = _metrics_by_name(metric_factories)

        for sample, image_path in zip(loader, calibration_images, strict=True):
            _, labels = sample
            source_predictions = parser.parse(
                ordered_outputs(reference.infer(image_path), output_names),
                class_map=class_map,
                **case.parser_params,
            )
            converted_predictions = parser.parse(
                ordered_outputs(
                    inferer.infer({stage.inputs[0].name: image_path}),
                    output_names,
                ),
                class_map=class_map,
                **case.parser_params,
            )
            for metric in source_metrics.values():
                metric.update(source_predictions, labels, **metric_ctx)
            for metric in converted_metrics.values():
                metric.update(converted_predictions, labels, **metric_ctx)

        source_results = _metric_results(source_metrics)
        converted_results = _metric_results(converted_metrics)
        assert_quality(
            source_results,
            converted_results,
            floors=case.budget.source_floors,
            max_drops=case.budget.max_drops[platform],
            case_id=case.id,
            platform=platform,
        )
    finally:
        shutil.rmtree(OUTPUTS_DIR / output_name, ignore_errors=True)
        shutil.rmtree(OUTPUTS_DIR / f"{output_name}_infer", ignore_errors=True)
