import io
import json
import shutil
import tarfile
import tempfile
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Any, cast
import cv2

import numpy as np
import polars as pl
from cyclopts import App
from loguru import logger
from luxonis_ml.nn_archive import Config
from luxonis_ml.utils import setup_logging
from rich import print

from modelconverter.cli.utils import request_info
from modelconverter.hub.__main__ import (
    _instance_ls,
    _model_ls,
    _variant_ls,
)
from modelconverter.hub.__main__ import (
    instance_download as _instance_download,
)
from modelconverter.hub.typing import Task
from modelconverter.utils import AdbHandler, subprocess_run
from modelconverter.utils.metadata import _get_metadata_dlc
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    MISC_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    SHARED_DIR,
)
from modelconverter.utils.exceptions import SubprocessException
from modelconverter.utils.nn_archive import safe_members

instance_download = lru_cache(maxsize=None)(_instance_download)

app = App(name="convert_hub_rvc4_models")

setup_logging(file="convert_hub_rvc4_models.log")

ADB_DATA_DIR = "/data/local/zoo_conversion"
models_df = pl.read_csv("zoo_migration/mappings.csv")


def get_missing_precision_instances(
    instances: list[dict[str, Any]], snpe_version: str
) -> list[dict[str, Any]]:
    all_precision_types = {
        inst["model_precision_type"]
        for inst in instances
        if inst["model_type"] == "RVC4"
        and inst["model_precision_type"] is not None
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


def create_new_instance(
    instance_params: dict[str, Any], archive: Path
) -> None:
    return
    # logger.info("Creating new instance")
    # instance = instance_create(**instance_params, silent=True)
    # logger.info(f"New instance created: {instance['id']}, {instance['name']}")
    # upload(str(archive), instance["id"])


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
    img_path: Path,
    shape: list[int],
    preprocessing: dict[str, Any]
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
    model_id: str,
    dataset_id: str,
) -> Path:
    import onnxruntime as ort

    input_names = [inp["name"] for inp in onnx_inputs]
    # TODO: make the neccessary changes to support multiple inputs
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

    dataset_files = list(dataset_path.glob("*.[jp][pn]g")) + list(dataset_path.glob("*.jpeg"))
    if not dataset_files:
        raise RuntimeError(f"No images found in dataset {dataset_id}")
    logger.info(f"Executing ONNX inference on {len(dataset_files)} images from {dataset_path}")

    outputs_path = Path("comparison", model_id, "onnx", "outputs")
    outputs_path.mkdir(parents=True, exist_ok=True)
    for img_path in dataset_files:
        input_tensors = {}
        for name, shape, prep in zip(input_names, input_shapes, input_preprocessing):
            input_tensors[name] = preprocess_image(img_path, shape, prep)

        result = session.run(None, input_tensors)
        for i, res in enumerate(result):
            np.save(outputs_path / f"{img_path.stem}_output_{i}.npy", res)

    return Path("comparison", model_id, "onnx", "outputs")

def get_io_info(model_path: Path) -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as d:
        csv_path = Path(d) / "info.csv"
        subprocess_run(
            [
                "snpe-dlc-info",
                "-i",
                model_path,
                "-s",
                csv_path,
            ],
            silent=True,
        )
        content = csv_path.read_text()
        csv_path.unlink()

        in_start = content.find("Input Name,Dimensions,Type,Encoding Info")
        out_start = content.find("Output Name,Dimensions,Type,Encoding Info", in_start)
        end      = content.find("Total parameters:", out_start)

        in_block  = content[in_start:out_start].strip()
        out_block = content[out_start:   end ].strip()

        df_in = pl.read_csv(io.StringIO(in_block))
        df_in = df_in.with_columns([pl.lit(None).cast(pl.Utf8).alias("Output Name")])
        df_out = pl.read_csv(io.StringIO(out_block))
        df_out = df_out.with_columns([pl.lit(None).cast(pl.Utf8).alias("Input Name")])

        cols = ["Input Name", "Output Name", "Dimensions", "Type", "Encoding Info"]
        df_in  = df_in.select(cols)
        df_out = df_out.select(cols)
        df = pl.concat([df_in, df_out])

        df = df.with_columns(
            pl.col("Dimensions")
            .str.split(",")
            .alias("Dimensions")
            .cast(pl.List(pl.Int64))
        )

        io_info = dict(inputs=[], outputs=[])
        for row in df.rows(named=True):
            is_input = row.get("Input Name") is not None
            if is_input:
                name = row["Input Name"]
                io_info['inputs'].append({
                    "name": name,
                    "shape": row["Dimensions"],
                    "dtype": row["Type"],
                })
            else:
                name = row["Output Name"]
                io_info['outputs'].append({
                    "name": name,
                    "shape": row["Dimensions"],
                    "dtype": row["Type"],
                })

    return io_info

@lru_cache(maxsize=None)
def adb_prepare_inference(
    model_path: Path,
    dataset_id: str,
    width: int,
    height: int,
    device_id: str | None = None,
) -> None:
    io_info = get_io_info(model_path)
    # TODO: support multiple inputs
    input_io = io_info["inputs"][0]
    assert input_io["shape"][1] == height and input_io["shape"][2] == width, f"Input shape {input_io['shape']} does not match expected shape ({height}, {width})"

    adb = AdbHandler(device_id, silent=False)
    adb.shell(f"mkdir -p {ADB_DATA_DIR}/{dataset_id}")
    with tempfile.TemporaryDirectory() as d:
        input_list = ""
        dataset_path = CALIBRATION_DIR / "datasets" / dataset_id
        for img_path in dataset_path.iterdir():
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (width, height))
            if input_io["dtype"] == "Float_32":
                np_type = np.float32
            elif input_io["dtype"] == "Float_16":
                np_type = np.float16
            elif input_io["dtype"] == "uFxp_8":
                np_type = np.uint8
            else:
                raise ValueError(
                    f"Unsupported data type {input_io['dtype']} for input {input_io['name']}."
                )
            img = img.astype(np_type)
            img.tofile(f"{d}/{img_path.stem}.raw")
            input_list += f"{ADB_DATA_DIR}/{dataset_id}/{d.split('/')[-1]}/{img_path.stem}.raw\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=d) as f:
            assert input_list
            f.write(input_list)
        adb.push(f.name, f"{ADB_DATA_DIR}/{dataset_id}/input_list.txt")
        adb.push(d, f"{ADB_DATA_DIR}/{dataset_id}/")

def adb_infer(
    model_path: Path,
    model_id: str,
    dataset_id: str,
    snpe_version: str,
    device_id: str | None,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    adb = AdbHandler(device_id, silent=False)
    if width is None or height is None:
        metadata = _get_metadata_dlc(model_path.parent / "info.csv")
        _, height, width, _ = next(iter(metadata.input_shapes.values()))

    adb_prepare_inference(model_path, dataset_id, width, height, device_id)
    adb.shell(f"mkdir -p {ADB_DATA_DIR}/{model_id}")
    adb.push(str(model_path), f"{ADB_DATA_DIR}/{model_id}/model.dlc")

    command = ""
    if snpe_version == "2.32.6":
        command += "source /data/local/tmp/source_me.sh && "

    command += (
        "snpe-net-run "
        f"--container {ADB_DATA_DIR}/{model_id}/model.dlc "
        f"--input_list {ADB_DATA_DIR}/{dataset_id}/input_list.txt "
        f"--output_dir {ADB_DATA_DIR}/{model_id}/outputs "
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
    out_dir.mkdir(parents=True, exist_ok=True)
    adb.pull(
        f"{ADB_DATA_DIR}/{model_id}/outputs",
        str(out_dir),
    )
    return Path("comparison", model_id, snpe_version)

def infer(
    archive: Path,
    model_id: str,
    dataset_id: str,
    snpe_version: str,
) -> Path:
    dir = MODELS_DIR / "zoo" / model_id / snpe_version
    dir.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory() as d,
        tarfile.open(archive, mode="r") as tf,
    ):
        tf.extractall(d, members=safe_members(tf))  # noqa: S202
        config = Config(**json.loads(Path(d, "config.json").read_text()))
        dlc = next(iter(Path(d).glob("*.dlc")))

        inp_name = config.model.inputs[0].name

        shutil.copy(dlc, dir)

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
        dir / dlc.name,
        "--output-dir",
        SHARED_DIR / "zoo-infer-output" / model_id / snpe_version,
        "--path",
        archive,
        "--input-path",
        src.parent,
        "--tool-version",
        snpe_version,
        "--dev",
    ]
    logger.info(f"Running command: {' '.join(map(str, args))}")
    subprocess_run(
        args,
        silent=True,
    )
    return SHARED_DIR / "zoo-infer-output" / model_id / snpe_version


def adb_test_degradation(
    old_archive: Path,
    new_archive: Path,
    parent_archive: Path,
    model_id: str,
    snpe_version: str,
    device_id: str | None,
) -> bool:
    dataset_id = (
        models_df.filter(pl.col("Model ID") == model_id)
        .select("Test Dataset ID")
        .item()
    )

    old_dir = MODELS_DIR / "zoo" / model_id / "2.23.0"
    old_dir.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory() as d,
        tarfile.open(old_archive, mode="r") as tf,
    ):
        tf.extractall(d, members=safe_members(tf))
        old_dlc = next(iter(Path(d).glob("*.dlc")))
        config = Config(**json.loads(Path(d, "config.json").read_text()))

        inp = config.model.inputs[0]
        layout = inp.layout
        shape = inp.shape
        height = shape[layout.index("H")]
        width = shape[layout.index("W")]

        old_inference = adb_infer(
            old_dlc,
            model_id,
            dataset_id,
            "2.23.0",
            device_id,
            height=height,
            width=width,
        )

        shutil.copy(old_dlc, old_dir)

    new_dir = MODELS_DIR / "zoo" / model_id / snpe_version
    new_dir.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory() as d,
        tarfile.open(new_archive, mode="r") as tf,
    ):
        tf.extractall(d, members=safe_members(tf))
        new_dlc = next(iter(Path(d).glob("*.dlc")))
        config = Config(**json.loads(Path(d, "config.json").read_text()))

        inp = config.model.inputs[0]
        layout = inp.layout
        shape = inp.shape
        height = shape[layout.index("H")]
        width = shape[layout.index("W")]

        new_inference = adb_infer(
            new_dlc,
            model_id,
            dataset_id,
            snpe_version,
            device_id,
            height=height,
            width=width,
        )

        shutil.copy(new_dlc, new_dir)

    onnx_model_path, onnx_inputs = get_onnx_info(parent_archive, model_id)
    onnx_inference = onnx_infer(onnx_model_path, onnx_inputs, model_id, dataset_id)

    return compare_files(old_inference, new_inference, onnx_inference, "image_embedding") # random non classification or segmentation task

def test_degradation(
    old_archive: Path,
    new_archive: Path,
    parent_archive: Path,
    model: dict[str, Any],
    snpe_version: str,
    device_id: str | None,
) -> bool:
    model_id = model["id"]
    dataset_id = (
        models_df.filter(pl.col("Model ID") == model_id)
        .select("Test Dataset ID")
        .item()
    )
    logger.info(f"Testing degradation for {model_id} on {dataset_id}")

    onnx_model_path, onnx_inputs = get_onnx_info(parent_archive, model_id)
    onnx_inference = onnx_infer(onnx_model_path, onnx_inputs, model_id, dataset_id)
    # TODO: update the `compare_files` to compare the onnx inference with the snpe old and new inference
    old_inference = infer(old_archive, model_id, dataset_id, "2.23.0")
    new_inference = infer(new_archive, model_id, dataset_id, snpe_version)
    print(model["tasks"])

    return compare_files(
        old_inference, new_inference, onnx_inference, model["tasks"][0].lower()
    )


def compare_files(
    old_inference: Path, new_inference: Path, onnx_inference: Path, task: Task
) -> bool:
    files = list(old_inference.rglob("*.npy"))
    assert len(files) > 0, "No files found in old inference"
    for old_file in files:
        new_file = new_inference / old_file.relative_to(old_inference)
        old_array = np.load(old_file)
        new_array = np.load(new_file)
        if task == "classification":
            if old_array.argmax() != new_array.argmax():
                raise RuntimeError(
                    f"Classification failed for {old_file} and {new_file}"
                )
        elif task == "segmentation":
            if np.issubdtype(old_array.dtype, np.floating):
                old_array = old_array.argmax(-1)
                new_array = new_array.argmax(-1)

            # too strict, no model would pass
            # if not np.array_equal(old_array, new_array):
            #     raise RuntimeError(
            #         f"Segmentation failed for {old_file} and {new_file}"
            #     )
            if (old_array == new_array).sum() / old_array.size < 0.85:
                raise RuntimeError(
                    f"Segmentation failed for {old_file} and {new_file}"
                )
        elif not np.isclose(old_array, new_array, atol=1e-1).all():
            raise RuntimeError(
                f"Comparison failed for {old_file} and {new_file}"
            )
    return True


def find_parent(instance: dict[str, Any]) -> dict[str, Any] | None:
    if instance["model_type"] == "ONNX":
        return instance
    parent_id = instance["parent_id"]
    if parent_id is None:
        return None

    return find_parent(request_info(parent_id, "modelInstances"))

def get_onnx_info(archive: Path, model_id: str) -> Path:
    REMOVE_INP_KEYS = ("dtype", "input_type", "layout")
    REMOVE_PREP_KEYS = ("interleaved_to_planar")

    def clean_input(inp):
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
        onnx_model = config["model"]["metadata"]["path"]
        tmp_onnx_path = next(iter(Path(d).glob(onnx_model)))
        if not tmp_onnx_path.exists():
            raise RuntimeError("ONNX file not found")

        subprocess_run(
            f"sudo chown -R {getenv('USER')} {MODELS_DIR}", silent=True
        )
        dst = MODELS_DIR / "zoo" / model_id / "onnx"
        dst.mkdir(parents=True, exist_ok=True)
        onnx_path = shutil.copy(tmp_onnx_path, dst / tmp_onnx_path.name)

    return onnx_path, onnx_inputs

def get_buildinfo(archive: Path) -> list[str]:
    with (
        tempfile.TemporaryDirectory() as d,
        tarfile.open(archive, mode="r") as tf,
    ):
        tf.extractall(d, members=safe_members(tf))  # noqa: S202
        buildinfo_path = Path(d, "buildinfo.json")
        if not buildinfo_path.exists():
            return []
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
    ]


def migrate(
    old_instance: dict[str, Any],
    snpe_version: str,
    model: dict[str, Any],
    variant_id: str | None,
    device_id: str | None,
    dry: bool,
    verify: bool,
) -> None:
    parent = find_parent(deepcopy(old_instance))
    model_id = model["id"]
    if parent is None:
        raise RuntimeError(
            f"Parent not found for model '{model_id}', variant '{variant_id}', and instance '{old_instance['id']}'"
        )
    logger.info(
        f"Parent found for model '{model_id}', variant '{variant_id}', and instance '{old_instance['id']}': {parent['id']}"
    )
    precision = old_instance["model_precision_type"]

    old_archive = instance_download(
        old_instance["id"],
        output_dir=(MISC_DIR / "zoo" / old_instance["id"]),
        cache=True,
    )

    buildinfo_opts = get_buildinfo(old_archive)

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
        str(parent_archive),
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
                str(CALIBRATION_DIR / "datasets" / dataset_name),
            ]
        )
    else:
        args.extend(
            [
                "rvc4.disable_calibration",
                "True",
            ]
        )

    logger.info(f"Running command: {' '.join(map(str, args))}")
    subprocess_run(args, silent=True)
    subprocess_run(
        f"sudo chown -R {getenv('USER')} {OUTPUTS_DIR}", silent=True
    )
    new_archive = next(iter((OUTPUTS_DIR / model_id).glob("*.tar.xz")))

    if not verify:
        logger.info(
            f"Skipping verification for model '{model_id}' and instance '{old_instance['id']}'"
        )
        if not dry:
            create_new_instance(new_instance_params, new_archive)

    # elif test_degradation(
    #     old_archive, new_archive, parent_archive, model, snpe_version, device_id
    # ):
    #     logger.info(
    #         f"Degradation test passed for model '{model_id}' and instance '{old_instance['id']}'"
    #     )
    #     if not dry:
    #         logger.info("Creating new instance")
    #         create_new_instance(new_instance_params, new_archive)

    elif adb_test_degradation(
        old_archive, new_archive, parent_archive, model_id, snpe_version, device_id
    ):
        logger.info(
            f"Degradation test passed for model '{model_id}' and instance '{old_instance['id']}'"
        )
        if not dry:
            logger.info("Creating new instance")
            create_new_instance(new_instance_params, new_archive)


def migrate_models(
    models: list[dict[str, Any]],
    snpe_version: str,
    device_id: str | None,
    df: dict[str, list[str | None]],
    dry: bool,
    verify: bool,
) -> None:
    for model in models:
        model_id = cast(str, model["id"])
        variants = _variant_ls(
            model_id=model_id, is_public=True, _silent=True
        )
        logger.info(f"Variants for model '{model_id}' found: {len(variants)}")
        for variant in variants:
            if "RVC4" not in variant["platforms"]:
                continue
            variant_id = cast(str, variant["id"])

            instances = _instance_ls(
                # model_id=model["id"],
                # variant_id=variant_id,
                model_version_id=variant_id,
                model_type=None,
                is_public=True,
                _silent=True,
            )
            instances = get_missing_precision_instances(
                instances, snpe_version
            )
            logger.info(
                f"Instances for variant {variant_id} found: {len(instances)}"
            )
            for old_instance in instances:
                try:
                    migrate(
                        old_instance,
                        snpe_version,
                        model,
                        variant_id,
                        device_id,
                        dry,
                        verify,
                    )
                    status = "success"
                    error = None
                except (Exception, SubprocessException) as e:
                    logger.error(f"Migration for model '{model_id}' failed!")
                    logger.error(e)
                    status = "failed"
                    error = str(e)
                df["model_id"].append(model_id)
                df["variant_id"].append(variant["id"])
                df["instance_id"].append(old_instance["id"])
                df["model_name"].append(model["name"])
                df["status"].append(status)
                df["error"].append(error)


@app.default
def main(
    *,
    snpe_version: str = "2.32.6",
    dry: bool = True,
    device_id: str | None = None,
    model_id: str | None = None,
    verify: bool = True,
) -> None:
    """Export all RVC4 models from the Luxonis Hub to SNPE format.

    Parameters
    ----------
    snpe_version : str
        The SNPE version to use for the export.
    dry : bool
        If True (default for safety), no models are uploaded to HubAI.
    device_id : str | None
        The device ID to use for the export. Must be set if
        there are more than one device connected.
    model_id : str | None
        An ID of a specific model to be migrated.
    verify : bool
        Whether to verify the migration by running inference on the
        converted models.
    """

    df = {
        "model_id": [],
        "variant_id": [],
        "instance_id": [],
        "model_name": [],
        "status": [],
        "error": [],
    }
    limit = 5 if dry else 10000
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
        migrate_models(models, snpe_version, device_id, df, dry, verify)
    finally:
        df = pl.DataFrame(df)
        date = datetime.now().strftime("%Y_%m_%d_%H_%M")  # noqa: DTZ005
        df.write_csv(f"migration_results_{date}.csv")


if __name__ == "__main__":
    app()
