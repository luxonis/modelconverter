import json
import re
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from functools import cache
from os import getenv
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import polars as pl
from cyclopts import App
from loguru import logger
from luxonis_ml.nn_archive import Config
from luxonis_ml.utils import setup_logging
from rich import print
from rich.prompt import Prompt
from scipy.spatial.distance import cosine

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
from modelconverter.utils.nn_archive import safe_members
from modelconverter.utils.types import DataType, Encoding, ResizeMethod

date = datetime.now().strftime("%Y_%m_%d_%H_%M")  # noqa: DTZ005
app = App(name="convert_hub_rvc4_models")

setup_logging(file="convert_hub_rvc4_models.log")

ADB_DATA_DIR = "/data/local/zoo_conversion"
models_df = pl.read_csv("mappings.csv")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - cosine(a.flatten(), b.flatten())  # type: ignore


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


def get_instance_params(
    inst: dict[str, Any], parent: dict[str, Any], snpe_version: str
) -> dict[str, Any]:
    model_id = inst["model_id"]
    return {
        "name": inst["name"],
        "variant_id": inst["model_version_id"],
        "model_type": "RVC4",
        "parent_id": parent["id"],
        "hardware_parameters": {"snpe_version": snpe_version},
        "model_precision_type": inst["model_precision_type"],
        "quantization_data": models_df.filter(pl.col("Model ID") == model_id)
        .select("Quant. Dataset ID")
        .item(),
        "tags": inst["tags"],
        "input_shape": inst["input_shape"],
    }


def preprocess_image(
    img_path: Path, shape: list[int], preprocessing: dict[str, Any]
) -> np.ndarray:
    img = cv2.imread(str(img_path))

    height, width = shape[2], shape[3]
    img = cv2.resize(img, (width, height)).astype(np.float32)

    if preprocessing.get("reverse_channels", False):
        img = img[..., ::-1]

    mean = np.array(preprocessing.get("mean", [0, 0, 0]), dtype=np.float32)
    scale = np.array(preprocessing.get("scale", [1, 1, 1]), dtype=np.float32)
    img = (img - mean) / scale

    return img.transpose(2, 0, 1)[None, ...]


def onnx_infer(
    onnx_model_path: Path,
    onnx_inputs: list[dict[str, Any]],
    onnx_outputs: list[str],
    model_id: str,
    dataset_id: str,
) -> Path:
    import onnxruntime as ort

    input_names = [inp["name"] for inp in onnx_inputs]
    if len(input_names) != 1:
        raise RuntimeError(
            f"Only single input models are supported for now, got a model with {len(input_names)} inputs"
        )
    input_shapes = [inp["shape"] for inp in onnx_inputs]
    input_preprocessing = [inp["preprocessing"] for inp in onnx_inputs]

    session = ort.InferenceSession(str(onnx_model_path))
    logger.info(f"Loaded ONNX model from '{onnx_model_path}'")

    dataset_path = CALIBRATION_DIR / "datasets" / dataset_id
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset {dataset_id} not found in {dataset_path}"
        )

    dataset_files = list(dataset_path.glob("*.[jp][pn]g")) + list(
        dataset_path.glob("*.jpeg")
    )
    if not dataset_files:
        raise RuntimeError(f"No images found in dataset {dataset_id}")

    logger.info(
        f"Executing ONNX inference on {len(dataset_files)} images from {dataset_path}"
    )

    outputs_path = Path("comparison", model_id, "onnx", "outputs")
    if outputs_path.exists():
        shutil.rmtree(outputs_path)
    outputs_path.mkdir(parents=True, exist_ok=True)
    for img_path in dataset_files:
        input_tensors = {}
        for name, shape, prep in zip(
            input_names, input_shapes, input_preprocessing, strict=True
        ):
            input_tensors[name] = preprocess_image(img_path, shape, prep)

        result = session.run(onnx_outputs, input_tensors)
        for i, res in enumerate(result):
            name = onnx_outputs[i]
            (outputs_path / name).mkdir(parents=True, exist_ok=True)
            np.save(outputs_path / name / f"{img_path.stem}.npy", res)

    return outputs_path


@cache
def adb_prepare_inference(
    dataset_id: str,
    in_shape: tuple[int, int, int, int],
    encoding: Encoding,
    resize_method: ResizeMethod,
    device_id: str | None = None,
) -> None:
    adb = AdbHandler(device_id, silent=False)
    adb.shell(f"mkdir -p {ADB_DATA_DIR}/{dataset_id}")
    with tempfile.TemporaryDirectory() as d:
        input_list = ""
        dataset_path = CALIBRATION_DIR / "datasets" / dataset_id
        for img_path in dataset_path.iterdir():
            n, h, w, c = in_shape
            arr = read_image(
                img_path,
                shape=[n, c, h, w],
                encoding=encoding,
                resize_method=resize_method,
                data_type=DataType.FLOAT32,
                transpose=False,
            )
            arr.tofile(f"{d}/{img_path.stem}.raw")
            input_list += f"{ADB_DATA_DIR}/{dataset_id}/{d.split('/')[-1]}/{img_path.stem}.raw\n"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=d) as f:
            assert input_list
            f.write(input_list)
        adb.push(f.name, f"{ADB_DATA_DIR}/{dataset_id}/input_list.txt")
        adb.push(d, f"{ADB_DATA_DIR}/{dataset_id}/")


def _infer_adb(
    model_path: Path,
    archive: Path,
    model_id: str,
    dataset_id: str,
    snpe_version: str,
    inp_name: str,
    device_id: str | None,
) -> Path:
    mult_cfg, _, _ = get_configs(str(archive))
    adb = AdbHandler(device_id, silent=False)
    config = mult_cfg.get_stage_config(None)

    in_shapes = {inp.name: inp.shape for inp in config.inputs}
    out_shapes = {out.name: out.shape for out in config.outputs}
    resize_method = {
        inp.name: inp.calibration.resize_method
        if isinstance(inp.calibration, ImageCalibrationConfig)
        else ResizeMethod.RESIZE
        for inp in config.inputs
    }
    encoding = {
        inp.name: inp.encoding.to
        if isinstance(inp.calibration, ImageCalibrationConfig)
        else Encoding.BGR
        for inp in config.inputs
    }

    in_shape = in_shapes[inp_name]
    adb_prepare_inference(
        dataset_id,
        tuple(in_shape),  # type: ignore
        encoding[inp_name],
        resize_method[inp_name],
        device_id,
    )

    adb_workdir = f"{ADB_DATA_DIR}/{model_id}/{snpe_version}/"

    adb.shell(f"mkdir -p {adb_workdir}")

    def source(snpe_version: str) -> str:
        return f"source /data/local/tmp/source_me_{snpe_version}.sh"

    adb.push(model_path, f"{adb_workdir}/model.dlc")

    command = (
        f"{source(snpe_version)} && "
        "snpe-net-run "
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

    out_dir = Path("comparison", model_id, snpe_version)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    raw_out_dir = out_dir / "raw"
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    adb.pull(f"{adb_workdir}/outputs", raw_out_dir)

    npy_out_dir = out_dir / "npy"
    npy_out_dir.mkdir(parents=True, exist_ok=True)

    for p in raw_out_dir.rglob("*.raw"):
        logger.warning(f"Processing {p}")
        arr = np.fromfile(p, dtype=np.float32)

        out_shape = out_shapes[p.stem]
        assert out_shape is not None

        if len(out_shape) == 4:
            N, H, W, C = out_shape
            arr = arr.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            arr = arr.reshape(out_shape)

        img_index = int(p.parent.name.split("_")[-1]) + 1
        logger.warning(f"Image index: {img_index}")
        dest = npy_out_dir / p.stem
        dest.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Saving to {dest}")
        np.save(dest / f"image_{img_index}.npy", arr)
    return npy_out_dir


def _infer_modelconv(
    dlc: Path,
    archive: Path,
    model_id: str,
    dataset_id: str,
    snpe_version: str,
    inp_name: str,
    save_dir: Path,
) -> Path:
    src = SHARED_DIR / "zoo-inference" / model_id / inp_name
    src.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        CALIBRATION_DIR / "datasets" / dataset_id, src, dirs_exist_ok=True
    )
    args = [
        "modelconverter",
        "infer",
        "rvc4",
        "--model-path",
        save_dir / dlc.name,
        "--output-dir",
        SHARED_DIR / "zoo-infer-output" / model_id / snpe_version,
        "--path",
        archive,
        "--input-path",
        src.parent,
        "--tool-version",
        snpe_version,
    ]
    logger.info(f"Running command: {' '.join(map(str, args))}")
    subprocess_run(args, silent=True)
    return SHARED_DIR / "zoo-infer-output" / model_id / snpe_version


def infer(
    archive: Path,
    model_id: str,
    dataset_id: str,
    snpe_version: str,
    infer_mode: Literal["adb", "modelconv"],
    device_id: str | None,
) -> Path:
    chown(SHARED_DIR)
    dir = MODELS_DIR / "zoo" / model_id / snpe_version
    dir.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory() as d,
        tarfile.open(archive, mode="r") as tf,
    ):
        tf.extractall(d, members=safe_members(tf))  # noqa: S202
        config = Config(**json.loads(Path(d, "config.json").read_text()))
        model_path = next(iter(Path(d).glob("*.dlc")))

        inp_name = config.model.inputs[0].name

        shutil.copy(model_path, dir)

        if infer_mode == "adb":
            return _infer_adb(
                model_path,
                archive,
                model_id,
                dataset_id,
                snpe_version,
                inp_name,
                device_id,
            )
        if infer_mode == "modelconv":
            return _infer_modelconv(
                model_path,
                archive,
                model_id,
                dataset_id,
                snpe_version,
                inp_name,
                dir,
            )
        logger.error(f"Unknown inference mode: {infer_mode}")
        sys.exit(1)


def test_degradation(
    old_archive: Path,
    new_archive: Path,
    parent_archive: Path,
    model: dict[str, Any],
    snpe_version: str,
    device_id: str | None,
    metric: Literal["mae", "mse", "psnr", "cos"],
    infer_mode: Literal["adb", "modelconv"],
) -> tuple[float, float]:
    model_id = model["id"]
    dataset_id = (
        models_df.filter(pl.col("Model ID") == model_id)
        .select("Test Dataset ID")
        .item()
    )
    logger.info(f"Testing degradation for {model_id} on {dataset_id}")

    onnx_model_path, onnx_inputs, onnx_outputs = get_onnx_info(
        parent_archive, model_id
    )
    onnx_inference = onnx_infer(
        onnx_model_path, onnx_inputs, onnx_outputs, model_id, dataset_id
    )
    old_inference = infer(
        old_archive, model_id, dataset_id, "2.23.0", infer_mode, device_id
    )
    new_inference = infer(
        new_archive, model_id, dataset_id, snpe_version, infer_mode, device_id
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
    metric: Literal["mae", "mse", "psnr", "cos"],
) -> tuple[float, float]:
    print(old_inference, new_inference, onnx_inference)
    files = list(old_inference.rglob("*.npy"))
    assert len(files) > 0, "No files found in old inference"

    metric_func: Callable

    if metric == "mae":
        metric_func = lambda a, b: np.mean(np.abs(a - b))  # noqa: E731
    elif metric == "mse":
        metric_func = lambda a, b: np.mean((a - b) ** 2)  # noqa: E731
    elif metric == "psnr":
        metric_func = psnr
    elif metric == "cos":
        metric_func = cosine_similarity
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    scores_new_vs_onnx = []
    scores_old_vs_onnx = []

    for old_file in files:
        relative_path = old_file.relative_to(old_inference)
        new_file = new_inference / relative_path
        onnx_file = onnx_inference / relative_path

        if not new_file.exists() or not onnx_file.exists():
            continue

        old_array = np.load(old_file)
        new_array = np.load(new_file)
        onnx_array = np.load(onnx_file)

        scores_new_vs_onnx.append(metric_func(new_array, onnx_array))
        scores_old_vs_onnx.append(metric_func(old_array, onnx_array))

    old_score = np.mean(scores_old_vs_onnx)
    new_score = np.mean(scores_new_vs_onnx)

    if old_score == float("nan") or new_score == float("nan"):
        raise RuntimeError(
            f"Degradation test failed: old model has NaN {metric} score ({old_score}) or new model has NaN {metric} score ({new_score})"
        )

    if metric == "cos":
        if old_score > new_score:
            raise RuntimeError(
                f"Degradation test failed: old model has higher {metric}  ({old_score}) than new model ({new_score})"
            )
    elif old_score < new_score:
        raise RuntimeError(
            f"Degradation test failed: old model has lower {metric}  ({old_score}) than new model ({new_score})"
        )
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
    old_instance: dict[str, Any],
    snpe_version: str,
    model: dict[str, Any],
    variant_id: str | None,
    device_id: str | None,
    verify: bool,
    instances: list[dict[str, Any]],
    metric: Literal["mae", "mse", "psnr", "cos"],
    infer_mode: Literal["adb", "modelconv"],
    *,
    upload: bool = False,
) -> tuple[float, float]:
    parent = find_parent(deepcopy(old_instance))
    if parent is None:
        logger.warning(
            f"Parent not found for {old_instance['id']}. Attempting to guess it."
        )
        parent = guess_parent(old_instance, instances)

    model_id = model["id"]
    if parent is None:
        raise RuntimeError(
            f"Parent not found for model '{model_id}', variant '{variant_id}', and instance '{old_instance['id']}'"
        )
    logger.info(
        f"Parent found for model '{model_id}', variant '{variant_id}', and instance '{old_instance['id']}': {parent['id']}"
    )
    old_archive = instance_download(
        old_instance["id"],
        output_dir=(MISC_DIR / "zoo" / old_instance["id"]),
        cache=True,
    )

    buildinfo_opts, command_args = get_buildinfo(old_archive)

    precision = old_instance["model_precision_type"]
    if precision is None:
        logger.warning(
            f"Precision is None for model '{model_id}' "
            f"and instance '{old_instance['id']}'"
        )
        if "--input_list" in command_args["quantization_cmd"]:
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
            f"and instance '{old_instance['id']}'"
        )

    new_instance_params = get_instance_params(
        old_instance, parent, snpe_version
    )

    parent_archive = instance_download(
        parent["id"],
        output_dir=(MISC_DIR / "zoo" / parent["id"]),
        cache=True,
    )

    dataset_name: str = (
        models_df.filter(pl.col("Model ID") == model_id)
        .select("Quant. Dataset ID")
        .item()
    )
    args = [
        "modelconverter",
        "convert",
        "rvc4",
        "--path",
        parent_archive,
        "--output-dir",
        model_id,
        "--to",
        "nn_archive",
        "--tool-version",
        snpe_version,
        *buildinfo_opts,
    ]

    if dataset_name.endswith("Dataset from Hub"):
        dataset_name = f"{dataset_name.split()[0].lower()}_quantization_data"

    if precision == "INT8":
        args.extend(
            [
                "calibration.path",
                CALIBRATION_DIR / "datasets" / dataset_name,
            ]
        )
    else:
        args.extend(["rvc4.disable_calibration", "True"])

    logger.info(f"Running command: {' '.join(map(str, args))}")
    subprocess_run(args, silent=True)
    chown(SHARED_DIR)
    new_archive = next(iter((OUTPUTS_DIR / model_id).glob("*.tar.xz")))

    if not verify:
        logger.info(
            f"Skipping verification for model '{model_id}' and instance '{old_instance['id']}'"
        )
    else:
        old_score, new_score = test_degradation(
            old_archive,
            new_archive,
            parent_archive,
            model,
            snpe_version,
            device_id,
            metric,
            infer_mode,
        )
        logger.info(
            f"Degradation test passed for model '{model_id}' and instance '{old_instance['id']}'"
        )
        sign = ">=" if metric == "cos" else "<="
        logger.info(
            f"New model {metric}: {new_score} {sign} old model {metric}: {old_score}"
        )
    if upload:
        upload_new_instance(new_instance_params, new_archive)
    return old_score, new_score


def migrate_models(
    models: list[dict[str, Any]],
    snpe_version: str,
    device_id: str | None,
    df: dict[str, list[str | float | None]],
    verify: bool,
    metric: Literal["mae", "mse", "psnr", "cos"],
    infer_mode: Literal["adb", "modelconv"],
    *,
    upload: bool = False,
) -> None:
    for model in models:
        model_id = cast(str, model["id"])
        variants = _variant_ls(model_id=model_id, is_public=True, _silent=True)
        logger.info(f"Variants for model '{model_id}' found: {len(variants)}")
        for variant in variants:
            if "RVC4" not in variant["platforms"]:
                continue
            variant_id = cast(str, variant["id"])

            all_instances = _instance_ls(
                model_version_id=variant_id,
                model_type=None,
                is_public=True,
                _silent=True,
            )
            logger.info(
                f"Instances for variant {variant_id} found: {len(all_instances)}"
            )
            instances = get_missing_precision_instances(
                all_instances, snpe_version
            )
            for old_instance in instances:
                try:
                    old_score, new_score = _migrate_models(
                        old_instance,
                        snpe_version,
                        model,
                        variant_id,
                        device_id,
                        verify,
                        all_instances,
                        metric,
                        infer_mode,
                        upload=upload,
                    )
                    status = "success"
                    error = None
                except (Exception, SubprocessException) as e:
                    logger.exception(
                        f"Migration for model '{model_id}' failed!"
                    )
                    status = "failed"
                    error = str(e)
                    old_score = new_score = None
                df["model_id"].append(model_id)
                df["variant_id"].append(variant["id"])
                df["instance_id"].append(old_instance["id"])
                df["model_name"].append(model["name"])
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
    verify: bool = True,
    infer_mode: Literal["adb", "modelconv"] = "modelconv",
    metric: Literal["mae", "mse", "psnr", "cos"] = "cos",
    limit: int = 5,
    upload: bool = False,
    confirm_upload: bool = False,
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
    verify : bool
        Whether to verify the migration by running inference on the
        converted models.
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
    limit = 1000 if upload else limit

    if device_id is not None:
        result = subprocess_run("adb devices", silent=True)
        if result.returncode == 0:
            pattern = re.compile(r"^(\w+)\s+device$", re.MULTILINE)
            devices = pattern.findall(result.stdout.decode())
            if device_id not in devices:
                raise ValueError(
                    f"Device ID '{device_id}' not found in adb devices. "
                    f"Available devices: {', '.join(devices)}"
                )
        elif result.returncode != 0:
            logger.warning("Unable to verify device ID")

    df = {
        "model_id": [],
        "variant_id": [],
        "instance_id": [],
        "model_name": [],
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

    try:
        migrate_models(
            models,
            snpe_version,
            device_id,
            df,
            verify,
            metric,
            infer_mode,
            upload=upload,
        )
    finally:
        df = pl.DataFrame(df)
        path = Path("results", f"migration_results_{date}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(path)


if __name__ == "__main__":
    app()
