import shutil
import subprocess
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import pytest
from luxonis_ml.utils import setup_logging

from modelconverter.utils import download_from_remote, subprocess_run
from modelconverter.utils.calibration_data import _get_from_remote

from .test_packages.metrics import (
    Metric,
    MNISTMetric,
    ResnetMetric,
    YoloV6Metric,
)

MODEL_CONFIGS = chain(
    *[
        [
            (
                service,
                "mnist",
                "s3://luxonis-test-bucket/modelconverter-test/mnist_dataset.zip",
                MNISTMetric,
                ["input"],
                "onnx",
                "",
            ),
            (
                service,
                "resnet18",
                "s3://luxonis-test-bucket/modelconverter-test/imagenette-extra-small.zip",
                ResnetMetric,
                ["input.1"],
                "onnx",
                "",
            ),
            (
                service,
                "resnet18",
                "s3://luxonis-test-bucket/modelconverter-test/imagenette-extra-small.zip",
                ResnetMetric,
                ["input.1"],
                "ir",
                "",
            ),
            (
                service,
                "resnet18",
                "s3://luxonis-test-bucket/modelconverter-test/imagenette-extra-small.zip",
                ResnetMetric,
                ["input.1"],
                "archive",
                "",
            ),
            (
                service,
                "yolov6n",
                "s3://luxonis-test-bucket/modelconverter-test/coco128-full.zip",
                YoloV6Metric,
                ["images"],
                "onnx",
                "",
            ),
            (
                service,
                "resnet18",
                "s3://luxonis-test-bucket/modelconverter-test/imagenette-extra-small.zip",
                ResnetMetric,
                ["input.1"],
                "onnx",
                "rvc4.disable_calibration true rvc3.disable_calibration true",
            ),
        ]
        for service in [
            "rvc2",
            "rvc2_superblob",
            "rvc3",
            "rvc3_quant",
            "rvc3_non_quant",
            "rvc4",
            "rvc4_non_quant",
            "hailo",
        ]
    ]
)

setup_logging(use_rich=True)


def prepare_fixture(
    service, model_name, dataset_url, metric, input_names, extra_args
):
    @pytest.fixture(scope="session")
    def _fixture():
        return prepare(
            service=service,
            model_name=model_name,
            dataset_url=dataset_url,
            metric=metric,
            input_names=input_names,
            extra_args=extra_args,
        )

    return _fixture


for (
    service,
    model,
    url,
    metric,
    inputs,
    model_type,
    extra_args,
) in MODEL_CONFIGS:
    fixture_name = f"{service}_{model}_{model_type}_env"
    fixture_function = prepare_fixture(
        service, model, url, metric, inputs, extra_args
    )
    exec(f"{fixture_name} = fixture_function")


def prepare(
    service: str,
    model_name: str,
    dataset_url: str,
    metric: Type[Metric],
    input_names: List[str],
    model_type: str = "onnx",
    extra_args: str = "",
) -> Tuple[
    str,
    Path,
    Path,
    Dict[str, float],
    List[Path],
    Path,
    Optional[subprocess.CompletedProcess],
    str,
]:
    onnx_url = (
        f"s3://luxonis-test-bucket/modelconverter-test/{model_name}.onnx"
    )
    config_url = f"gs://luxonis-test-bucket/modelconverter/{model_name}.yaml"
    converted_model_path_prefix = (
        Path("shared_with_container") / "outputs" / f"_{model_name}-test"
    )
    if service in ["rvc4", "rvc4_non_quant"]:
        converted_model_path = (
            converted_model_path_prefix / f"{model_name}.dlc"
        )
    elif service in ["rvc2", "rvc3", "rvc3_non_quant"]:
        converted_model_path = (
            converted_model_path_prefix
            / "intermediate_outputs"
            / f"{model_name}-simplified.xml"
        )
    elif service == "rvc2_superblob":
        converted_model_path = (
            converted_model_path_prefix / f"{model_name}.superblob"
        )
    elif service == "rvc3_quant":
        converted_model_path = (
            converted_model_path_prefix
            / "intermediate_outputs"
            / f"{model_name}-simplified-int8.xml"
        )
    elif service == "hailo":
        converted_model_path = (
            converted_model_path_prefix / f"{model_name}.har"
        )
    else:
        raise NotImplementedError

    dataset_path = _get_from_remote(
        dataset_url, Path("tests/data/test_packages/datasets")
    )
    onnx_path = download_from_remote(
        onnx_url, "tests/data/test_packages/models"
    )
    if model_type == "ir":
        file_url = f"s3://luxonis-test-bucket/modelconverter-test/{model_name}-simplified.bin"
    elif model_type == "onnx":
        file_url = onnx_url
    else:
        file_url = f"s3://luxonis-test-bucket/modelconverter-test/archives/{model_name}.tar.xz"

    if "quant" not in service or "non_quant" in service:
        result_convert = subprocess_run(
            f"modelconverter convert {service.replace('_superblob', '').replace('_non_quant', '')} "
            f"--path {config_url} "
            "--dev "
            "--no-gpu "
            f"input_model {file_url} "
            f"output_dir_name _{model_name}-test "
            "hailo.compression_level 0 "
            "hailo.optimization_level 0 "
            "hailo.early_stop True "
            f"rvc2.superblob {'false' if 'superblob' not in service else 'true'} "
            "calibration.max_images 30 "
            + ("--to nn_archive" if model_type == "archive" else "")
            + " "
            + extra_args,
        )
    else:
        result_convert = None
    expected_metric = metric.eval_onnx(onnx_path, dataset_path)

    for name, value in expected_metric.items():
        assert value > 0.7, f"{name} is too low: {value}"

    input_files_dirs = [
        Path("shared_with_container")
        / "inference_inputs"
        / f"_{model_name}-test"
        / input_name
        for input_name in input_names
    ]
    for input_files_dir in input_files_dirs:
        if input_files_dir.exists():
            shutil.rmtree(input_files_dir)
        input_files_dir.mkdir(parents=True)

    dest = (
        Path("shared_with_container")
        / "inference_outputs"
        / f"_{model_name}-test"
    )
    dest.mkdir(parents=True, exist_ok=True)
    return (
        config_url,
        converted_model_path,
        dataset_path,
        expected_metric,
        input_files_dirs,
        dest,
        result_convert,
        service.replace("_quant", "").replace("_non", ""),
    )
