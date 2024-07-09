import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from modelconverter.utils import subprocess_run

from .metrics import MNISTMetric, ResnetMetric, YoloV6Metric

TOLERANCE: float = 0.05


def compare_metrics(
    metrics: Dict[str, float], expected_metrics: Dict[str, float]
):
    for metric, value in metrics.items():
        assert value == pytest.approx(expected_metrics[metric], abs=TOLERANCE)


def check_convert(convert_env):
    *_, result, _ = convert_env
    assert result.returncode == 0


def mnist_infer(mnist_env):
    (
        config_url,
        converted_model_path,
        dataset_path,
        expected_metric,
        (input_files_dir, *_),
        dest,
        _,
        service,
    ) = mnist_env

    for img_path in dataset_path.iterdir():
        shutil.copy(img_path, input_files_dir)

    result = subprocess_run(
        f"modelconverter infer {service} "
        f"--model-path {converted_model_path} "
        f"--dest {dest} "
        f"--input-path {input_files_dir.parent} "
        f"--path {config_url} "
        "--dev "
        "--no-gpu"
    )
    assert result.returncode == 0, result.stderr + result.stdout

    metric = MNISTMetric()

    for output_path in (dest / "output").iterdir():
        output = np.load(output_path)
        label = int(output_path.stem.split("_")[-1])
        metric.update(output, label)

    compare_metrics(metric.get_result(), expected_metric)


def resnet18_infer(resnet18_env):
    (
        config_url,
        converted_model_path,
        dataset_path,
        expected_metric,
        (input_files_dir, *_),
        dest,
        _,
        service,
    ) = resnet18_env

    for label in dataset_path.iterdir():
        for img_path in label.iterdir():
            shutil.copy(
                img_path,
                input_files_dir
                / f"{img_path.stem}_{label.stem}{img_path.suffix}",
            )

    result = subprocess_run(
        f"modelconverter infer {service} "
        f"--model-path {converted_model_path} "
        f"--dest {dest} "
        f"--input-path {input_files_dir.parent} "
        f"--path {config_url} "
        "--dev "
        "--no-gpu"
    )
    assert result.returncode == 0, result.stderr + result.stdout

    metric = ResnetMetric()

    for output_path in (dest / "191").iterdir():
        output = np.load(output_path)
        label = int(output_path.stem.split("_")[-1])
        metric.update(output, label)

    compare_metrics(metric.get_result(), expected_metric)


def yolov6n_infer(yolov6n_env):
    output_names = [f"output{i}_yolov6r2" for i in range(1, 4)]
    (
        config_url,
        converted_model_path,
        dataset_path,
        expected_metric,
        (input_files_dir, *_),
        dest,
        _,
        service,
    ) = yolov6n_env

    labels = {}

    for img_path in (dataset_path / "images" / "train2017").iterdir():
        label_path = str(img_path.with_suffix(".txt")).replace(
            "images", "labels"
        )
        if not Path(label_path).exists():
            continue
        labels[img_path.stem] = YoloV6Metric.read_label(label_path)
        shutil.copy(img_path, input_files_dir)

    result = subprocess_run(
        f"modelconverter infer {service} "
        f"--model-path {converted_model_path} "
        f"--dest {dest} "
        f"--input-path {input_files_dir.parent} "
        f"--path {config_url} "
        "--dev "
        "--no-gpu"
    )
    assert result.returncode == 0, result.stderr + result.stdout

    metric = YoloV6Metric()

    for output1, output2, output3 in zip(
        (dest / output_names[0]).iterdir(),
        (dest / output_names[1]).iterdir(),
        (dest / output_names[2]).iterdir(),
    ):
        output = [
            np.load(output1),
            np.load(output2),
            np.load(output3),
        ]
        label = labels[output1.stem]
        metric.update(output, label)

    compare_metrics(metric.get_result(), expected_metric)
