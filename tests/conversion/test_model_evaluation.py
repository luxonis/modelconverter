"""Task-metric regression tests for real public HubAI source models.

Unlike ``test_precision``, these score predictions against labelled images.
Luxonis Eval is used purely as a test library -- modelconverter still does the
conversion and runs the native artifact through its own inferers -- so no Luxonis
device is needed.

Run in an RVC2/RVC4 development image::

    modelconverter shell rvc2 --dev -c \\
      'python -m pytest tests/conversion/test_model_evaluation.py -m rvc2'
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from modelconverter.__main__ import convert
from modelconverter.cli.utils import get_configs
from modelconverter.packages.getters import get_inferer
from modelconverter.utils.constants import CALIBRATION_DIR, OUTPUTS_DIR
from modelconverter.utils.general import sanitize_net_name
from modelconverter.utils.types import Target
from tests.helpers.evaluation import assert_quality, ordered_outputs
from tests.helpers.onnx_reference import ONNXReferenceInferer
from tests.helpers.precision import locate_converted_model
from tests.helpers.target_options import target_options

COCO_SAMPLE = "gs://luxonis-test-bucket/luxonis-ml-test-data/coco_sample.zip"
DATASET_NAME = "modelconverter-coco-sample-evaluation"
CALIBRATION_SPEC = f"{DATASET_NAME}:train"
IMAGE_SHAPE = (288, 512)

# Linked calibration for yolov8-seg's `coeffs`: pick the highest-scoring cell
# across the three detection heads and take its mask coefficients.
YOLOV8_COEFFS_SCRIPT = """
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
class EvaluationCase:
    id: str
    # An immutable HubAI source instance, not a mutable model slug.
    instance_id: str
    parser: str
    parser_params: dict[str, Any]
    metrics: tuple[str, ...]
    # Floors reject broken source artifacts; the per-platform drops cap how much
    # COCO-sample AP a converted model may lose against that source output.
    floors: dict[str, float]
    max_drops: dict[str, dict[str, float]]


CASES = (
    EvaluationCase(
        id="yolov6n-detection",
        instance_id="aimi_LCEFX2rJSsMhEjsyeMEcWn",
        parser="detection",
        parser_params={"subtype": "yolov6r2", "n_classes": 80},
        metrics=("bbox",),
        floors={"bbox.AP": 0.14, "bbox.AP50": 0.22},
        max_drops={
            "rvc2": {"bbox.AP": 0.03, "bbox.AP50": 0.03},
            "rvc4": {"bbox.AP": 0.04, "bbox.AP50": 0.04},
        },
    ),
    EvaluationCase(
        id="yolov8n-instance-segmentation",
        instance_id="aimi_QtcY6CKM2cxB2QpmHVU4QL",
        parser="instance_segmentation",
        parser_params={"subtype": "yolov8", "n_classes": 80},
        metrics=("bbox", "mask"),
        floors={
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
)


@pytest.fixture(scope="module")
def coco_sample(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Parse the shared COCO sample GCS archive into an LDF."""
    from luxonis_ml.data import LuxonisLoader, LuxonisParser
    from luxonis_ml.utils.environ import environ

    work_dir = tmp_path_factory.mktemp("modelconverter-eval-fixture")
    old_base_path = environ.LUXONISML_BASE_PATH
    environ.LUXONISML_BASE_PATH = work_dir / "luxonis_ml"
    height, width = IMAGE_SHAPE
    dataset = LuxonisParser(
        COCO_SAMPLE,
        dataset_name=DATASET_NAME,
        delete_local=True,
        save_dir=work_dir / "parsed",
    ).parse()
    try:
        yield LuxonisLoader(
            dataset,
            view=["train", "val", "test"],
            height=height,
            width=width,
            keep_aspect_ratio=False,
            color_space="BGR",
        )
    finally:
        dataset.delete_dataset(delete_local=True)
        shutil.rmtree(CALIBRATION_DIR / DATASET_NAME, ignore_errors=True)
        environ.LUXONISML_BASE_PATH = old_base_path


def _calibration_images() -> list[Path]:
    """The images materialized by modelconverter's LDF loader."""
    directory = CALIBRATION_DIR / DATASET_NAME
    paths = sorted(directory.glob("*.png"), key=lambda path: int(path.stem))
    assert paths, f"no calibration images under {directory}"
    return paths


def _link_options(stage: str, post_stage: str, inputs: list[str]) -> list[str]:
    """Overrides wiring yolov8-seg's postprocessor to its upstream stage."""
    prototypes = inputs.index("prototypes")
    coeffs = inputs.index("coeffs")
    prefix = f"stages.{post_stage}.inputs"
    return [
        f"{prefix}.{prototypes}.name", "prototypes",
        f"{prefix}.{prototypes}.calibration.stage", stage,
        f"{prefix}.{prototypes}.calibration.output", "protos_output",
        f"{prefix}.{coeffs}.name", "coeffs",
        f"{prefix}.{coeffs}.calibration.stage", stage,
        f"{prefix}.{coeffs}.calibration.script", YOLOV8_COEFFS_SCRIPT,
    ]  # fmt: skip


def _results(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        f"{name}.{key}": float(value)
        for name, metric in metrics.items()
        for key, value in metric.compute().items()
        if key != "metric"
    }


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


@pytest.mark.parametrize(("platform", "case"), _PARAMS)
def test_real_model_task_metrics(
    platform: str, case: EvaluationCase, tmp_path: Path, coco_sample: Any
):
    """Converted native outputs retain source quality on labelled COCO data."""
    # Imported here, not at module level: the rvc3/hailo jobs collect this module
    # too, and only the rvc2/rvc4 dev images carry `requirements-eval.txt`.
    from hubai_sdk import HubAIClient
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

    parser = {
        "detection": YOLODetectionParser,
        "instance_segmentation": YOLOInstanceSegmentationParser,
    }[case.parser]()
    metric_types = {
        "bbox": BboxMeanAveragePrecision,
        "mask": MaskMeanAveragePrecision,
    }

    target = Target(platform)
    options = target_options(target)
    archive = HubAIClient().instances.download_instance(
        case.instance_id, str(tmp_path / "models")
    )

    cfg, _, _ = get_configs(target, str(archive), list(options))
    stage_name, stage = next(iter(cfg.stages.items()))
    # The reference inferer and the metric context both assume NCHW at the
    # fixture's geometry.
    assert len(stage.inputs) == 1
    assert stage.inputs[0].layout == "NCHW"
    assert (stage.inputs[0].shape or [])[2:] == list(IMAGE_SHAPE)

    convert_options = [
        *options,
        f"stages.{stage_name}.calibration.path",
        CALIBRATION_SPEC,
        f"stages.{stage_name}.calibration.max_images",
        "16",
    ]
    if case.parser == "instance_segmentation":
        post_name, post_stage = next(
            (name, config)
            for name, config in cfg.stages.items()
            if name != stage_name
        )
        convert_options += _link_options(
            stage_name, post_name, [inp.name for inp in post_stage.inputs]
        )

    output_name = sanitize_net_name(f"eval_{platform}_{case.id}")
    try:
        convert(
            target,
            *convert_options,
            path=str(archive),
            output_dir=output_name,
            to="native",
        )
        images = _calibration_images()
        assert len(images) == len(coco_sample)

        model_path = locate_converted_model(
            OUTPUTS_DIR / output_name, platform
        )
        reference = ONNXReferenceInferer.from_stage(stage)
        inferer = get_inferer(
            target,
            str(model_path),
            images[0].parent,
            OUTPUTS_DIR / f"{output_name}_infer",
            stage,
        )

        output_names = [output.name for output in stage.outputs]
        ldf_class_map = {
            index: name for name, index in coco_sample._classes[""].items()
        }
        class_map = get_dataset_class_mapping("coco")
        height, width = IMAGE_SHAPE
        metric_ctx = get_metric_ctx(
            {},
            width=width,
            height=height,
            ldf_class_map=ldf_class_map,
            class_map=class_map,
            class_index_map=get_class_index_mapping(ldf_class_map, class_map),
        )
        source = {name: metric_types[name]() for name in case.metrics}
        converted = {name: metric_types[name]() for name in case.metrics}

        for sample, image in zip(coco_sample, images, strict=True):
            _, labels = sample
            for metrics, outputs in (
                (source, reference.infer(image)),
                (converted, inferer.infer({stage.inputs[0].name: image})),
            ):
                predictions = parser.parse(
                    ordered_outputs(outputs, output_names),
                    class_map=class_map,
                    **case.parser_params,
                )
                for metric in metrics.values():
                    metric.update(predictions, labels, **metric_ctx)

        assert_quality(
            _results(source),
            _results(converted),
            floors=case.floors,
            max_drops=case.max_drops[platform],
            case_id=case.id,
            platform=platform,
        )
    finally:
        shutil.rmtree(OUTPUTS_DIR / output_name, ignore_errors=True)
        shutil.rmtree(OUTPUTS_DIR / f"{output_name}_infer", ignore_errors=True)
