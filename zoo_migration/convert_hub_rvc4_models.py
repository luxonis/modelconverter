import json
import math
import re
import shutil
import signal
import sys
import tarfile
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from os import getenv
from pathlib import Path
from types import FrameType
from typing import Any, Literal, NoReturn, cast

import cv2
import numpy as np
import polars as pl
from cyclopts import App
from loguru import logger
from luxonis_ml.nn_archive import Config
from luxonis_ml.utils import setup_logging
from metric import Metric
from onnxruntime import InferenceSession
from rich.prompt import Prompt

from modelconverter.cli.utils import get_configs, request_info
from modelconverter.hub.__main__ import (
    _instance_ls,
    _model_ls,
    _variant_ls,
    instance_create,
    instance_download,
    upload,
)
from modelconverter.utils import AdbHandler, read_image, subprocess_run
from modelconverter.utils.config import ImageCalibrationConfig
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    MISC_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    SHARED_DIR,
)
from modelconverter.utils.exceptions import SubprocessException
from modelconverter.utils.metadata import get_metadata
from modelconverter.utils.nn_archive import safe_members
from modelconverter.utils.types import DataType, Encoding, ResizeMethod

date = datetime.now().strftime("%Y_%m_%d_%H_%M")  # noqa: DTZ005
app = App(name="convert_hub_rvc4_models")

setup_logging(file=f"results/convert_hub_rvc4_models_{date}.log")

ADB_DATA_DIR = Path("/data/local/zoo_conversion/datasets")
ADB_MODELS_DIR = Path("/data/local/zoo_conversion/models")
models_df = pl.read_csv("mappings.csv")


@dataclass
class InputFileMapping:
    files_by_input: dict[str, dict[str, Path]]
    common_indices: set[str]


def validate_and_map_input_files(
    dataset_path: Path, input_names: list[str]
) -> InputFileMapping:
    if len(input_names) == 1:
        return InputFileMapping({}, set())

    files_by_input: dict[str, dict[str, Path]] = {}
    index_sets: dict[str, set[str]] = {}

    file_patterns = ("*.jpg", "*.jpeg", "*.png", "*.npy")

    for name in input_names:
        subdir = dataset_path / name
        if not subdir.is_dir():
            raise FileNotFoundError(
                f"Expected folder for input '{name}' at {subdir}"
            )

        all_files = []
        for pattern in file_patterns:
            all_files.extend(subdir.glob(pattern))

        if not all_files:
            raise RuntimeError(
                f"No files found in '{subdir}' for input '{name}'"
            )

        key_to_path: dict[str, Path] = {}
        indices: set[str] = set()

        for file_path in sorted(all_files):
            match = re.search(r"_(\d+)$", file_path.stem)
            if not match:
                match = re.search(r"(\d+)$", file_path.stem)

            if not match:
                raise RuntimeError(
                    f"Cannot extract index from filename '{file_path.name}'"
                )

            idx = match.group(1)
            if idx in indices:
                raise RuntimeError(
                    f"Duplicate index '{idx}' in folder '{subdir}'"
                )

            indices.add(idx)
            key_to_path[idx] = file_path

        files_by_input[name] = key_to_path
        index_sets[name] = indices

    common_indices = set.intersection(*index_sets.values())
    for name, indices in index_sets.items():
        if indices != common_indices:
            missing = common_indices - indices
            extra = indices - common_indices
            raise RuntimeError(
                f"Index mismatch for input '{name}': "
                f"missing {missing or 'none'}, extra {extra or 'none'}"
            )

    return InputFileMapping(files_by_input, common_indices)


def chown(path: Path) -> None:
    subprocess_run(f"sudo chown -R {getenv('USER')} {path}", silent=True)


def get_missing_precision_instances(
    instances: list[dict[str, Any]], snpe_version: str
) -> list[dict[str, Any]]:
    all_precision_types = {
        inst["model_precision_type"]
        for inst in instances
        if inst["model_type"] == "RVC4"
    }
    snpe_version_precision_types = {
        inst["model_precision_type"]
        for inst in instances
        if inst["model_type"] == "RVC4"
        and (inst["hardware_parameters"] or {}).get("snpe_version")
        == snpe_version
    }
    missing = all_precision_types - snpe_version_precision_types
    return [
        inst
        for inst in instances
        if inst["model_type"] == "RVC4"
        and inst["model_precision_type"] in missing
    ]


def upload_new_instance(
    instance_params: dict[str, Any], archive: Path
) -> None:
    logger.info("Creating new instance")
    instance = instance_create(**instance_params, silent=True)
    logger.info(f"New instance created: {instance['id']}, {instance['name']}")
    upload(str(archive), instance["id"])


def filter_models_df(model_id: str, variant_id: str | None) -> pl.DataFrame:
    df = models_df.filter(pl.col("Model ID") == model_id)
    if df["Variant ID"].drop_nulls().len() > 0 and variant_id is not None:
        df = df.filter(pl.col("Variant ID") == variant_id)
    return df


def get_instance_params(
    inst: dict[str, Any],
    variant_id: str,
    parent: dict[str, Any],
    snpe_version: str,
) -> dict[str, Any]:
    model_id = inst["model_id"]
    filtered_df = filter_models_df(model_id, variant_id)
    return {
        "name": inst["name"],
        "variant_id": inst["model_version_id"],
        "model_type": "RVC4",
        "parent_id": parent["id"],
        "hardware_parameters": {"snpe_version": snpe_version},
        "model_precision_type": inst["model_precision_type"],
        "quantization_data": filtered_df.select("Quant. Dataset ID").item(),
        "tags": inst["tags"],
        "input_shape": inst["input_shape"],
    }


def preprocess_image(
    img_path: Path,
    shape: list[int],
    dtype: np.dtype,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    channels, height, width = shape[1], shape[2], shape[3]

    if channels == 1:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (width, height)).astype(dtype)
        img = img[..., np.newaxis]
        mean_default = [0.0]
        scale_default = [1.0]
    else:
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (width, height)).astype(dtype)
        if preprocessing.get("reverse_channels", False):
            img = img[..., ::-1]
        mean_default = [0.0, 0.0, 0.0]
        scale_default = [1.0, 1.0, 1.0]

    mean = np.array(preprocessing.get("mean", mean_default), dtype=dtype)
    scale = np.array(preprocessing.get("scale", scale_default), dtype=dtype)
    img = (img - mean) / scale

    return img.transpose(2, 0, 1)[None, ...]


def onnx_infer(
    onnx_model_path: Path,
    onnx_inputs: list[dict[str, Any]],
    onnx_outputs: list[str],
    model_id: str,
    variant_id: str,
    instance_id: str,
    dataset_id: str,
) -> Path:
    import onnxruntime as ort

    dtypes_map = {
        "float16": np.float16,
        "float32": np.float32,
        "float": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "int8": np.int8,
    }

    input_names = [inp["name"] for inp in onnx_inputs]
    input_shapes = [inp["shape"] for inp in onnx_inputs]
    input_dtypes = [inp["dtype"] for inp in onnx_inputs]
    input_preprocessing = [inp["preprocessing"] for inp in onnx_inputs]

    session = ort.InferenceSession(str(onnx_model_path))
    logger.info(f"Loaded ONNX model from '{onnx_model_path}'")

    dataset_path = CALIBRATION_DIR / "datasets" / dataset_id
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset {dataset_id} not found in {dataset_path}"
        )
    logger.info(f"Using dataset at '{dataset_path}'")

    outputs_path = Path(
        "comparison", model_id, variant_id, instance_id, "onnx", "outputs"
    )
    if outputs_path.exists():
        shutil.rmtree(outputs_path)
    outputs_path.mkdir(parents=True, exist_ok=True)

    mapping = validate_and_map_input_files(dataset_path, input_names)

    def run_onnx(
        session: InferenceSession,
        out_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> None:
        results = session.run(onnx_outputs, inputs)
        for out_name, arr in zip(out_names, results, strict=True):
            out_dir = outputs_path / out_name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"{idx}.npy", arr)

    if len(mapping.common_indices) == 0:
        file_patterns = ["*.[jp][pn]g", "*.jpeg", "*.npy"]
        files = []

        for pattern in file_patterns:
            files.extend(dataset_path.glob(pattern))

        if not files:
            raise RuntimeError(f"No files found in dataset at {dataset_path}")

        for file_path in sorted(files):
            name = input_names[0]
            shape = input_shapes[0]
            dtype = dtypes_map[input_dtypes[0]]
            prep = input_preprocessing[0]
            idx = file_path.stem.split("_")[-1]

            tensor = preprocess_image(file_path, shape, dtype, prep)
            run_onnx(session, onnx_outputs, {name: tensor})

    else:
        for idx in sorted(mapping.common_indices, key=lambda x: int(x)):
            input_files = {
                name: mapping.files_by_input[name][idx] for name in input_names
            }

            input_tensors = {}

            for name, shape, dtype, prep in zip(
                input_names,
                input_shapes,
                input_dtypes,
                input_preprocessing,
                strict=True,
            ):
                file_path = input_files[name]
                if file_path.suffix.lower() == ".npy":
                    tensor = np.load(file_path).astype(dtypes_map[dtype])
                else:
                    tensor = preprocess_image(
                        file_path, shape, dtypes_map[dtype], prep
                    )
                input_tensors[name] = tensor
            run_onnx(session, onnx_outputs, input_tensors)

    return outputs_path


def adb_prepare_inference(
    dataset_id: str,
    input_names: list[str],
    in_shapes: dict[str, tuple[int, int, int, int]],
    encodings: dict[str, Encoding],
    resize_methods: dict[str, ResizeMethod],
    device_id: str | None = None,
) -> None:
    adb = AdbHandler(device_id, silent=False)
    adb.shell(f"mkdir -p {ADB_DATA_DIR}/{dataset_id}")

    dataset_path = CALIBRATION_DIR / "datasets" / dataset_id

    mapping = validate_and_map_input_files(dataset_path, input_names)

    with tempfile.TemporaryDirectory() as tmpdir:
        list_lines: list[str] = []

        if len(mapping.common_indices) == 0:
            file_patterns = ("*.jpg", "*.jpeg", "*.png", "*.npy")
            files: list[Path] = []
            for pattern in file_patterns:
                files.extend(dataset_path.glob(pattern))

            if not files:
                raise RuntimeError(f"No files found in dataset {dataset_id}")

            for file_path in sorted(files):
                name = input_names[0]
                shape = in_shapes[name]
                enc = encodings[name]
                rm = resize_methods[name]

                n, *s, c = shape
                arr = read_image(
                    file_path,
                    shape=[n, c, *s],
                    encoding=enc,
                    resize_method=rm,
                    data_type=DataType.FLOAT32,
                    transpose=False,
                ).astype(np.float32)

                raw_name = f"{file_path.stem}.raw"
                host_raw = Path(tmpdir) / raw_name
                arr.tofile(host_raw)
                list_lines.append(
                    f"{ADB_DATA_DIR}/{dataset_id}/{tmpdir.split('/')[-1]}/{raw_name}"
                )
        else:
            for idx in sorted(mapping.common_indices, key=lambda x: int(x)):
                input_files = {
                    name: mapping.files_by_input[name][idx]
                    for name in input_names
                }

                parts: list[str] = []

                for name in input_names:
                    file_path = input_files[name]
                    n, *s, c = in_shapes[name]
                    arr = read_image(
                        file_path,
                        shape=[n, c, *s],
                        encoding=encodings[name],
                        resize_method=resize_methods[name],
                        data_type=DataType.FLOAT32,
                        transpose=False,
                    ).astype(np.float32)

                    raw_name = f"{name}_{idx}.raw"
                    host_path = Path(tmpdir) / raw_name
                    arr.tofile(host_path)

                    dev_path = f"{ADB_DATA_DIR}/{dataset_id}/{tmpdir.split('/')[-1]}/{raw_name}"
                    parts.append(f"{name}:={dev_path}")

                list_lines.append(" ".join(parts))

        input_list_path = Path(tmpdir) / "input_list.txt"
        input_list_path.write_text("\n".join(list_lines) + "\n")
        adb.push(input_list_path, ADB_DATA_DIR / dataset_id / "input_list.txt")
        input_list_path.unlink()
        adb.push(tmpdir, ADB_DATA_DIR / dataset_id)


def adb_infer(
    model_path: Path,
    archive: Path,
    model_id: str,
    variant_id: str,
    instance_id: str,
    dataset_id: str,
    snpe_version: str,
    device_id: str | None,
) -> Path:
    logger.info(f"Running inference on ADB. Parsing archive {archive}...")
    mult_cfg, _, _ = get_configs(str(archive))
    adb = AdbHandler(device_id, silent=False)
    config = mult_cfg.get_stage_config(None)

    in_names = [inp.name for inp in config.inputs]
    in_shapes = {inp.name: inp.shape for inp in config.inputs}
    out_shapes = {out.name: out.shape for out in config.outputs}
    resize_method = {
        inp.name: inp.calibration.resize_method
        if isinstance(inp.calibration, ImageCalibrationConfig)
        else ResizeMethod.RESIZE
        for inp in config.inputs
    }
    encoding = {
        inp.name: (
            Encoding.GRAY
            if inp.encoding.to == Encoding.GRAY
            else (
                inp.encoding.to
                if isinstance(inp.calibration, ImageCalibrationConfig)
                else Encoding.BGR
            )
        )
        for inp in config.inputs
    }

    adb_prepare_inference(
        dataset_id,
        in_names,
        in_shapes,  # type: ignore
        encoding,
        resize_method,
        device_id,
    )
    out_dir = Path(
        "comparison", model_id, variant_id, instance_id, snpe_version
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)

    adb_workdir = (
        ADB_MODELS_DIR / model_id / variant_id / instance_id / snpe_version
    )

    adb.shell(f"mkdir -p {adb_workdir}")

    def source(snpe_version: str) -> str:
        return f"source /data/local/tmp/source_me_{snpe_version}.sh"

    adb.push(model_path, adb_workdir / "model.dlc")

    adb.shell(
        f"{source(snpe_version)} && "
        f"snpe-dlc-info "
        f"-i {adb_workdir}/model.dlc "
        f"-s {adb_workdir}/model_info.csv"
    )
    adb.pull(adb_workdir / "model_info.csv", out_dir / "model_info.csv")
    metadata = get_metadata(out_dir / "model_info.csv")

    command = (
        f"{source(snpe_version)} && "
        f"snpe-net-run "
        f"--container {adb_workdir}/model.dlc "
        f"--input_list {ADB_DATA_DIR}/{dataset_id}/input_list.txt "
        f"--output_dir {adb_workdir}/outputs "
        "--perf_profile default "
        "--enable_cpu_fallback "
        "--use_dsp"
    )
    ret, stdout, stderr = adb.shell(command)

    if ret != 0:
        raise SubprocessException(
            f"SNPE inference failed with code {ret}:\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}\n"
        )

    raw_out_dir = out_dir / "raw"
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    adb.pull(adb_workdir / "outputs", raw_out_dir)

    npy_out_dir = out_dir / "npy"
    npy_out_dir.mkdir(parents=True, exist_ok=True)

    for p in raw_out_dir.rglob("*.raw"):
        logger.debug(f"Processing {p}")
        arr = np.fromfile(p, dtype=np.float32)

        out_shape = out_shapes[p.stem]
        assert out_shape is not None

        arr = arr.reshape(metadata.output_shapes[p.stem])

        img_index = int(p.parent.name.split("_")[-1]) + 1
        dest = npy_out_dir / p.stem
        dest.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving to {dest}")
        np.save(dest / f"image_{img_index}.npy", arr)

    return npy_out_dir


def _infer_modelconv(
    dlc: Path,
    archive: Path,
    model_id: str,
    variant_id: str,
    instance_id: str,
    dataset_id: str,
    snpe_version: str,
    inp_names: list[str],
    save_dir: Path,
) -> Path:
    if len(inp_names) == 1:
        src = (
            SHARED_DIR
            / "zoo-inference"
            / model_id
            / variant_id
            / instance_id
            / inp_names[0]
        )
        src.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            CALIBRATION_DIR / "datasets" / dataset_id, src, dirs_exist_ok=True
        )
    else:
        for inp_name in inp_names:
            src = (
                SHARED_DIR
                / "zoo-inference"
                / model_id
                / variant_id
                / instance_id
                / inp_name
            )
            src.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                CALIBRATION_DIR / "datasets" / dataset_id / inp_name,
                src,
                dirs_exist_ok=True,
            )

    args = [
        "modelconverter",
        "infer",
        "rvc4",
        "--model-path",
        save_dir / dlc.name,
        "--output-dir",
        SHARED_DIR
        / "zoo-infer-output"
        / model_id
        / variant_id
        / instance_id
        / snpe_version,
        "--path",
        archive,
        "--input-path",
        src.parent,
        "--tool-version",
        snpe_version,
    ]
    logger.info(f"Running command: {' '.join(map(str, args))}")
    subprocess_run(args, silent=True)
    return (
        SHARED_DIR
        / "zoo-infer-output"
        / model_id
        / variant_id
        / instance_id
        / snpe_version
    )


def infer(
    archive: Path,
    model_id: str,
    variant_id: str,
    instance_id: str,
    dataset_id: str,
    snpe_version: str,
    infer_mode: Literal["adb", "modelconv"],
    device_id: str | None,
) -> Path:
    chown(SHARED_DIR)
    dir = (
        MODELS_DIR / "zoo" / model_id / variant_id / instance_id / snpe_version
    )
    dir.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory() as d,
        tarfile.open(archive, mode="r") as tf,
    ):
        tf.extractall(d, members=safe_members(tf))  # noqa: S202
        config = Config(**json.loads(Path(d, "config.json").read_text()))
        model_path = next(iter(Path(d).glob("*.dlc")))

        inp_names = [inp.name for inp in config.model.inputs]

        shutil.copy(model_path, dir)

        if infer_mode == "adb":
            return adb_infer(
                model_path,
                archive,
                model_id,
                variant_id,
                instance_id,
                dataset_id,
                snpe_version,
                device_id,
            )
        if infer_mode == "modelconv":
            return _infer_modelconv(
                model_path,
                archive,
                model_id,
                variant_id,
                instance_id,
                dataset_id,
                snpe_version,
                inp_names,
                dir,
            )
        logger.error(f"Unknown inference mode: {infer_mode}")
        sys.exit(1)


def test_degradation(
    old_archive: Path,
    new_archive: Path,
    parent_archive: Path,
    model: dict[str, Any],
    variant_id: str,
    instance_id: str,
    snpe_version: str,
    device_id: str | None,
    metric: Metric,
    infer_mode: Literal["adb", "modelconv"],
) -> tuple[float, float]:
    model_id = model["id"]
    filtered_df = filter_models_df(model_id, variant_id)
    dataset_id = filtered_df.select("Test Dataset ID").item()
    logger.info(f"Testing degradation for {model_id} on {dataset_id}")

    onnx_model_path, onnx_inputs, onnx_outputs = get_onnx_info(
        parent_archive, model_id
    )
    onnx_inference = onnx_infer(
        onnx_model_path,
        onnx_inputs,
        onnx_outputs,
        model_id,
        variant_id,
        instance_id,
        dataset_id,
    )
    old_inference = infer(
        old_archive,
        model_id,
        variant_id,
        instance_id,
        dataset_id,
        "2.23.0",
        infer_mode,
        device_id,
    )
    new_inference = infer(
        new_archive,
        model_id,
        variant_id,
        instance_id,
        dataset_id,
        snpe_version,
        infer_mode,
        device_id,
    )

    return compare_files(
        old_inference, new_inference, onnx_inference, metric=metric
    )


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")  # Perfect match
    max_pixel = np.max([a.max(), b.max()])
    return 20 * np.log10(max_pixel) - 10 * np.log10(mse)


def compare_files(
    old_inference: Path,
    new_inference: Path,
    onnx_inference: Path,
    metric: Metric,
) -> tuple[float, float]:
    files = list(old_inference.rglob("*.npy"))
    assert len(files) > 0, "No files found in old inference"

    scores_new_vs_onnx = []
    scores_old_vs_onnx = []

    for old_file in files:
        relative_path = old_file.relative_to(old_inference)
        idx = int(old_file.stem.split("_")[-1])
        new_file = new_inference / relative_path
        onnx_file = (onnx_inference / relative_path).parent / f"{idx}.npy"

        if not new_file.exists() or not onnx_file.exists():
            raise ValueError(
                f"Some of the inferred files do not exists: {new_file}: {new_file.exists()}, {onnx_file}: {onnx_file.exists()}"
            )

        old_array = np.load(old_file)
        new_array = np.load(new_file)
        onnx_array = np.load(onnx_file)

        scores_new_vs_onnx.append(metric.compute(new_array, onnx_array))
        scores_old_vs_onnx.append(metric.compute(old_array, onnx_array))

    if not scores_new_vs_onnx or not scores_old_vs_onnx:
        raise RuntimeError(f"No scores computed for metric {metric.value}. ")

    old_score = float(np.mean(scores_old_vs_onnx))
    new_score = float(np.mean(scores_new_vs_onnx))

    if math.isnan(old_score) or math.isnan(new_score):
        raise RuntimeError(
            f"Degradation test failed: old model has NaN {metric.value} score ({old_score}) or new model has NaN {metric.value} score ({new_score})"
        )

    if math.isclose(old_score, new_score, rel_tol=5e-2, abs_tol=1e-5):
        return old_score, new_score  # type: ignore

    metric.verify(old_score, new_score)
    return old_score, new_score  # type: ignore


def find_parent(instance: dict[str, Any]) -> dict[str, Any] | None:
    if instance["model_type"] == "ONNX":
        return instance
    parent_id = instance["parent_id"]
    if parent_id is None:
        return None

    return find_parent(request_info(parent_id, "modelInstances"))


def get_onnx_info(
    archive: Path, model_id: str
) -> tuple[Path, list[dict[str, Any]], list[str]]:
    REMOVE_INP_KEYS = ("dtype", "input_type", "layout")
    REMOVE_PREP_KEYS = "interleaved_to_planar"

    def clean_input(inp: dict[str, Any]) -> dict[str, Any]:
        cleaned = {k: v for k, v in inp.items() if k not in REMOVE_INP_KEYS}
        prep = cleaned.get("preprocessing", {})
        cleaned["preprocessing"] = {
            k: v for k, v in prep.items() if k not in REMOVE_PREP_KEYS
        }
        return cleaned

    def get_input_dtype(
        onnx_path: Path, inputs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))

        input_name_to_meta = {inp.name: inp for inp in session.get_inputs()}

        updated_inputs = []
        for inp_dict in inputs:
            name = inp_dict.get("name")
            if name in input_name_to_meta:
                onnx_input_meta = input_name_to_meta[name]
                dtype_str = onnx_input_meta.type
                if dtype_str.startswith("tensor(") and dtype_str.endswith(")"):
                    dtype = dtype_str[len("tensor(") : -1]
                else:
                    logger.warning(
                        f"Unexpected ONNX input type format: {dtype_str} for input {name}"
                    )
                    dtype = (
                        "float32"  # Default to float32 if format is unexpected
                    )

                inp_dict_copy = inp_dict.copy()
                inp_dict_copy["dtype"] = dtype
                updated_inputs.append(inp_dict_copy)
            else:
                raise RuntimeError(
                    f"Input '{name}' from config not found in ONNX model inputs."
                )
        return updated_inputs

    with (
        tempfile.TemporaryDirectory() as d,
        tarfile.open(archive, mode="r") as tf,
    ):
        tf.extractall(d, members=safe_members(tf))  # noqa: S202
        config_path = Path(d, "config.json")
        config = json.loads(config_path.read_text())
        if not config_path.exists():
            raise RuntimeError("Config file not found")
        onnx_inputs = [clean_input(inp) for inp in config["model"]["inputs"]]
        output_names = [outp["name"] for outp in config["model"]["outputs"]]
        onnx_model = config["model"]["metadata"]["path"]
        tmp_onnx_path = next(iter(Path(d).glob(onnx_model)))
        if not tmp_onnx_path.exists():
            raise RuntimeError("ONNX file not found")
        onnx_inputs = get_input_dtype(tmp_onnx_path, onnx_inputs)

        chown(SHARED_DIR)
        dst = MODELS_DIR / "zoo" / model_id / "onnx"
        dst.mkdir(parents=True, exist_ok=True)
        onnx_path = shutil.copy(tmp_onnx_path, dst / tmp_onnx_path.name)

    return Path(onnx_path), onnx_inputs, output_names


def get_buildinfo(archive: Path) -> tuple[list[str], dict[str, Any]]:
    with (
        tempfile.TemporaryDirectory() as d,
        tarfile.open(archive, mode="r") as tf,
    ):
        tf.extractall(d, members=safe_members(tf))  # noqa: S202
        buildinfo_path = Path(d, "buildinfo.json")
        if not buildinfo_path.exists():
            logger.warning(f"Buildinfo file not found in {archive}.")
            return [], {}
        buildinfo = json.loads(buildinfo_path.read_text())

    if "modelconverter_version" not in buildinfo:
        raise NotImplementedError("Multi-stage archive")

    buildinfo = buildinfo["cmd_info"]

    def remove_args(command: list[str], to_remove: list[str]) -> list[str]:
        result = []
        i = 0
        while i < len(command):
            if command[i] in to_remove:
                # Skip the flag and its associated value
                i += 2
            else:
                result.append(command[i])
                i += 1
        return result

    def jsonify(lst: list[str]) -> str:
        string = json.dumps(lst, separators=(",", ":"))
        # ' and " needs to be swapped for the CLI
        return string.replace("'", "@@").replace('"', "'").replace("@@", '""')

    convert_args = remove_args(buildinfo["dlc_convert"][1:], ["-i"])

    if "quantization_cmd" in buildinfo:
        quant_args = remove_args(
            buildinfo["quantization_cmd"][1:],
            ["--input_list", "--input_dlc", "--output_dlc"],
        )
    else:
        quant_args = []

    graph_args = remove_args(
        buildinfo["graph_prepare"][1:], ["--input_dlc", "--output_dlc"]
    )

    return [
        "rvc4.snpe_onnx_to_dlc_args",
        jsonify(convert_args),
        "rvc4.snpe_dlc_quant_args",
        jsonify(quant_args),
        "rvc4.snpe_dlc_graph_prepare_args",
        jsonify(graph_args),
    ], buildinfo


def guess_parent(
    orphan: dict[str, Any], instances: list[dict[str, Any]]
) -> dict[str, Any] | None:
    born = datetime.fromisoformat(orphan["created"])
    suspected_parents = [
        inst
        for inst in instances
        if inst["id"] != orphan["id"]
        and inst["model_type"] == "ONNX"
        and datetime.fromisoformat(inst["created"]) < born
    ]
    suspected_parents.sort(
        key=lambda x: datetime.fromisoformat(x["created"]), reverse=True
    )
    return suspected_parents[0] if suspected_parents else None


def _migrate_models(
    *,
    old_instance: dict[str, Any],
    parent: dict[str, Any] | None,
    snpe_version: str,
    model: dict[str, Any],
    variant_id: str,
    device_id: str | None,
    verify: bool,
    metric: Metric,
    infer_mode: Literal["adb", "modelconv"],
    upload: bool = False,
    skip_conversion: bool = False,
) -> tuple[float, float]:
    old_instance_id = old_instance["id"]

    model_id = model["id"]
    if parent is None:
        raise RuntimeError(
            f"Parent not found for model '{model_id}', variant '{variant_id}', and instance '{old_instance_id}'"
        )
    logger.info(
        f"Parent found for model '{model_id}', variant '{variant_id}', and instance '{old_instance_id}': {parent['id']}"
    )
    old_archive = instance_download(
        old_instance_id,
        output_dir=(
            MISC_DIR / "zoo" / model_id / variant_id / old_instance_id
        ),
        cache=True,
    )

    buildinfo_opts, command_args = get_buildinfo(old_archive)

    precision = old_instance["model_precision_type"]
    if precision is None:
        logger.warning(
            f"Precision is `None` for model '{model_id}' "
            f"and instance '{old_instance_id}'"
        )
        if "quantization_cmd" not in command_args:
            precision = "FP32"
        elif "--input_list" in command_args["quantization_cmd"]:
            precision = "INT8"
        elif "--float_bitwidth" in command_args["quantization_cmd"]:
            i = command_args["quantization_cmd"].index("--float-bitwidth")
            if command_args["quantization_cmd"][i + 1] == "16":
                precision = "FP16"
            else:
                precision = "FP32"
        else:
            precision = "FP32"
        logger.warning(
            f"Precision guessed as '{precision}' for model '{model_id}' "
            f"and instance '{old_instance_id}'"
        )

    new_instance_params = get_instance_params(
        old_instance, variant_id, parent, snpe_version
    )

    parent_archive = instance_download(
        parent["id"],
        output_dir=(MISC_DIR / "zoo" / model_id / variant_id / parent["id"]),
        cache=True,
    )

    filtered_df = filter_models_df(model_id, variant_id)
    dataset_name: str = filtered_df.select("Quant. Dataset ID").item()
    args = [
        "modelconverter",
        "convert",
        "rvc4",
        "--path",
        parent_archive,
        "--output-dir",
        f"{model_id}/{variant_id}/{old_instance_id}_new",
        "--to",
        "nn_archive",
        "--tool-version",
        snpe_version,
        *buildinfo_opts,
    ]

    if dataset_name.endswith("Dataset from Hub"):
        dataset_name = f"{dataset_name.split()[0].lower()}_quantization_data"

    if precision == "INT8":
        calibration_path = CALIBRATION_DIR / "datasets" / dataset_name
        args.extend(["calibration.path", calibration_path])
        if calibration_path.exists() and dataset_name != "/":
            logger.info(
                f"Using calibration dataset at {calibration_path} for model '{model_id}' and instance '{old_instance_id}'"
            )
        else:
            raise FileNotFoundError(
                f"Calibration dataset at {calibration_path} not found for model '{model_id}' and instance '{old_instance_id}'"
            )
    else:
        args.extend(["rvc4.disable_calibration", "True"])

    outdir = OUTPUTS_DIR / model_id / variant_id / f"{old_instance_id}_new"
    if skip_conversion and outdir.exists() and any(outdir.glob("*.tar.xz")):
        logger.info(
            f"Skipping conversion for model '{model_id}', variant {variant_id}, and instance '{old_instance_id}'"
        )
    else:
        logger.info(f"Running command: {' '.join(map(str, args))}")
        subprocess_run(args, silent=True)
        chown(SHARED_DIR)

    new_archive = next(iter((outdir).glob("*.tar.xz")))

    if not verify:
        logger.info(
            f"Skipping verification for model '{model_id}' and instance '{old_instance_id}'"
        )
    else:
        old_score, new_score = test_degradation(
            old_archive,
            new_archive,
            parent_archive,
            model,
            variant_id,
            old_instance_id,
            snpe_version,
            device_id,
            metric,
            infer_mode,
        )
        logger.info(
            f"Degradation test passed for model '{model_id}' and instance '{old_instance_id}'"
        )
        logger.info(
            f"New model {metric.value}: {new_score} {metric.sign} old model {metric.value}: {old_score}"
        )
    if upload:
        upload_new_instance(new_instance_params, new_archive)
    return old_score, new_score


def migrate_models(
    *,
    models: list[dict[str, Any]],
    snpe_version: str,
    device_id: str | None,
    df: dict[str, list[str | float | None]],
    verify: bool,
    metric: Metric,
    infer_mode: Literal["adb", "modelconv"],
    upload: bool = False,
    skip_conversion: bool = False,
    variant_id: str | None = None,
    instance_id: str | None = None,
) -> None:
    for model in models:
        model_id = cast(str, model["id"])
        variants = _variant_ls(model_id=model_id, is_public=True, _silent=True)
        logger.info(f"Variants for model '{model_id}' found: {len(variants)}")
        for variant in variants:
            if "RVC4" not in variant["platforms"]:
                continue
            version_id = cast(str, variant["id"])
            if variant_id is not None and version_id != variant_id:
                continue

            all_instances = _instance_ls(
                model_version_id=version_id,
                model_type=None,
                is_public=True,
                _silent=True,
            )
            logger.info(
                f"Instances for variant {version_id} found: {len(all_instances)}"
            )
            instances = get_missing_precision_instances(
                all_instances, snpe_version
            )
            for old_instance in instances:
                if (
                    instance_id is not None
                    and old_instance["id"] != instance_id
                ):
                    continue
                old_score = new_score = None

                parent = find_parent(deepcopy(old_instance))
                if parent is None:
                    logger.warning(
                        f"Parent not found for {old_instance['id']}. Attempting to guess it."
                    )
                    parent = guess_parent(old_instance, all_instances)

                try:
                    old_score, new_score = _migrate_models(
                        old_instance=old_instance,
                        parent=parent,
                        snpe_version=snpe_version,
                        model=model,
                        variant_id=version_id,
                        device_id=device_id,
                        verify=verify,
                        metric=metric,
                        infer_mode=infer_mode,
                        upload=upload,
                        skip_conversion=skip_conversion,
                    )
                    if math.isclose(
                        old_score, new_score, rel_tol=1e-3, abs_tol=1e-5
                    ):
                        status = "success"
                    else:
                        status = "passable"
                    error = None
                except (Exception, SubprocessException) as e:
                    logger.exception(
                        f"Migration for model '{model_id}' failed!"
                    )
                    status = "failed"
                    error = str(e)
                df["model_id"].append(model_id)
                df["variant_id"].append(variant["id"])
                df["instance_id"].append(old_instance["id"])
                df["parent_id"].append(parent["id"] if parent else None)
                df["model_name"].append(model["name"])
                df["precision"].append(old_instance["model_precision_type"])
                df["status"].append(status)
                df["error"].append(error)
                df["old_to_onnx_score"].append(old_score)
                df["new_to_onnx_score"].append(new_score)


@app.default
def main(
    *,
    snpe_version: str = "2.32.6",
    device_id: str | None = None,
    model_id: str | None = None,
    variant_id: str | None = None,
    instance_id: str | None = None,
    infer_mode: Literal["adb", "modelconv"] = "modelconv",
    metric: Metric = Metric.MSE,
    limit: int = 5,
    upload: bool = False,
    confirm_upload: bool = False,
    skip_conversion: bool = False,
    verify: bool = True,
) -> None:
    """Export all RVC4 models from the Luxonis Hub to SNPE format.

    Parameters
    ----------
    snpe_version : str
        The SNPE version to use for the export.
    device_id : str | None
        The device ID to use for the export. Must be set if
        there are more than one device connected.
    model_id : str | None
        An ID of a specific model to be migrated.
    variant_id : str | None
        An ID of a specific variant to be migrated. If set, the `model_id` must also be set.
    instance_id : str | None
        An ID of a specific instance to be migrated. If set, the `variant_id` and `model_id` must also be set.
    infer_mode : Literal["adb", "modelconv"]
        The inference mode to use. "modelconv" does not require a device,
        but the results can be misleading due to de-quantized CPU inference.
    metric : Literal["mae", "mse", "psnr", "cos"]
        The metric to use for the degradation test.
    limit : int
        The maximum number of models to process. Default is 5 for safety.
        Ignored when `--upload` is set, unless `--model-id` is also set.
    upload : bool
        If True, the converted models are uploaded to HubAI.
    confirm_upload : bool
        Needs to be set together with `--upload` for extra safety.
    skip_conversion : bool
        If True, the conversion step is skipped and the existing
        converted models are used. This is useful for debugging.
    verify : bool
        Whether to verify the migration by running inference on the
        converted models.
    """
    if upload ^ confirm_upload:
        raise ValueError(
            "To upload the converted models to production zoo, you must "
            "set both --upload and --confirm-upload flags to prevent "
            "accidental modification of production data."
        )
    if upload and confirm_upload:
        confirmation = Prompt.ask(
            "Are you sure you want to upload the converted "
            "models to production zoo?",
            choices=["yes", "no"],
        )
        if confirmation != "yes":
            logger.info("Upload cancelled")
            sys.exit(0)
    if variant_id is not None and model_id is None:
        raise ValueError(
            "If --variant-id is set, --model-id must also be set."
        )
    if instance_id is not None and variant_id is None:
        raise ValueError(
            "If --instance-id is set, --variant-id must also be set."
        )
    limit = 1000 if upload else limit

    if infer_mode == "adb":
        result = subprocess_run("adb devices", silent=True)
        if result.returncode == 0:
            pattern = re.compile(r"^(\w+)\s+device$", re.MULTILINE)
            devices = pattern.findall(result.stdout.decode())
        else:
            raise RuntimeError("Unable to verify device ID")

        if device_id is None:
            if len(devices) == 0:
                raise RuntimeError("No devices connected")
            logger.warning(
                "No device ID specified, using the first connected "
                f"device: {devices[0]}"
            )

            device_id = devices[0]
        elif device_id not in devices:
            raise ValueError(
                f"Device ID '{device_id}' not found in connected devices: {devices}"
            )
        else:
            logger.info(f"Using device ID: {device_id}")

    df = {
        "model_id": [],
        "variant_id": [],
        "instance_id": [],
        "parent_id": [],
        "model_name": [],
        "precision": [],
        "status": [],
        "error": [],
        "old_to_onnx_score": [],
        "new_to_onnx_score": [],
    }
    if model_id is not None:
        models = [request_info(model_id, "models")]
    else:
        models = _model_ls(
            is_public=True,
            luxonis_only=True,
            limit=limit,
            _silent=True,
        )
    logger.info(f"Models found: {len(models)}")

    def cleanup() -> None:
        nonlocal df
        pl_df = pl.DataFrame(df)
        path = Path("results", f"migration_results_{date}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        pl_df.write_csv(path)
        logger.info(f"Results saved to {path}")
        n_success = pl_df.filter(
            pl.col("status").is_in(["success", "passable"])
        ).shape[0]
        logger.info(
            f"Migration completed. {n_success} out of {len(pl_df)} models "
            f"were successfully migrated."
        )

    def signal_handler(signum: int, frame: FrameType | None) -> NoReturn:
        logger.error(f"Received signal {signum}, cleaning up before exiting.")
        cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        migrate_models(
            models=models,
            snpe_version=snpe_version,
            device_id=device_id,
            df=df,
            verify=verify,
            metric=metric,
            infer_mode=infer_mode,
            upload=upload,
            skip_conversion=skip_conversion,
            variant_id=variant_id,
            instance_id=instance_id,
        )
    finally:
        cleanup()


if __name__ == "__main__":
    app()
