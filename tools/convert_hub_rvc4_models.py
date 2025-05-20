import json
import tarfile
import tempfile
from copy import deepcopy
from functools import lru_cache
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
from modelconverter.utils import AdbHandler, subprocess_run
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    MISC_DIR,
    OUTPUTS_DIR,
)
from modelconverter.utils.exceptions import SubprocessException
from modelconverter.utils.metadata import _get_metadata_dlc
from modelconverter.utils.nn_archive import safe_members
from modelconverter.utils.types import DataType

instance_download = lru_cache(maxsize=None)(_instance_download)

app = App(name="convert_hub_rvc4_models")

setup_logging(file="convert_hub_rvc4_models.log")

ADB_DATA_DIR = "/data/local/zoo_conversion/"
df = pl.read_csv("models.csv")

models_df = {"original": [], "parent": []}


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


def get_instance_params(
    inst: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    model_id = inst["model_id"]
    return {
        "model_version_id": inst["model_version_id"],
        "model_type": "RVC4",
        "parent_id": parent["id"],
        "model_precision_type": inst["model_precision_type"],
        "quantization_data": df.filter(pl.col("Model ID") == model_id)
        .select("Quant. Dataset ID")
        .item(),
        "tags": inst["tags"],
        "input_shape": inst["input_shape"],
    }


def test_degradation(
    old_archive: Path,
    new_dlc: Path,
    model_id: str,
    snpe_version: str,
    device_id: str | None,
) -> bool:
    dataset_id = (
        df.filter(pl.col("Model ID") == model_id)
        .select("Test Dataset ID")
        .item()
    )

    with (
        tempfile.TemporaryDirectory() as d,
        tarfile.open(old_archive, mode="r") as tf,
    ):
        tf.extractall(d, members=safe_members(tf))  # noqa: S202
        old_dlc = next(iter(Path(d).glob("*.dlc")))
        config = Config(**json.loads(Path(d, "config.json").read_text()))

        inp = config.model.inputs[0]
        layout = inp.layout
        shape = inp.shape
        height = shape[layout.index("H")]
        width = shape[layout.index("W")]

        old_inference = infer(
            old_dlc,
            model_id,
            dataset_id,
            snpe_version,
            device_id,
            height=height,
            width=width,
            data_type=DataType(inp.dtype.value),
        )
        print(old_inference)
    new_inference = infer(new_dlc, model_id, dataset_id, snpe_version)
    print(new_inference)
    return compare_files(old_inference, new_inference)


def compare_files(old_inference: Path, new_inference: Path) -> bool:
    for old_file in old_inference.rglob("*.raw"):
        new_file = new_inference / old_file.relative_to(old_inference)

        print(old_file)
        print(new_file)
        old_array = np.fromfile(old_file, dtype=np.float32)
        new_array = np.fromfile(new_file, dtype=np.float32)
        if not np.isclose(old_array, new_array).all():
            logger.error(
                f"Degradation test failed for {old_file} and {new_file}"
            )
            return False
    return True


@lru_cache
def prepare_inference(
    dataset_id: str,
    width: int,
    height: int,
    data_type: DataType,
    device_id: str | None = None,
) -> None:
    adb = AdbHandler(device_id)
    adb.shell(f"mkdir -p {ADB_DATA_DIR}/{dataset_id}")
    with tempfile.TemporaryDirectory() as d:
        input_list = ""
        for img_path in Path("datasets", dataset_id).iterdir():
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (width, height))
            img = img.astype(data_type.as_numpy_dtype())
            img.tofile(f"{d}/{img_path.stem}.raw")
            input_list += f"{ADB_DATA_DIR}/{dataset_id}/{img_path.stem}.raw\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=d) as f:
            f.write(input_list)
            adb.push(f.name, f"{ADB_DATA_DIR}/{dataset_id}/input_list.txt")
        adb.push(d, f"{ADB_DATA_DIR}/{dataset_id}/")


def infer(
    model_path: Path,
    model_id: str,
    dataset_id: str,
    snpe_version: str,
    device_id: str | None = None,
    width: int | None = None,
    height: int | None = None,
    data_type: DataType | None = None,
) -> Path:
    adb = AdbHandler(device_id)
    if width is None or height is None:
        metadata = _get_metadata_dlc(model_path.parent / "info.csv")
        _, height, width, _ = next(iter(metadata.input_shapes.values()))
        data_type = next(iter(metadata.input_dtypes.values()))
    if data_type is None:
        metadata = _get_metadata_dlc(model_path.parent / "info.csv")
        data_type = next(iter(metadata.input_dtypes.values()))

    prepare_inference(dataset_id, width, height, data_type, device_id)
    adb.shell(f"mkdir -p {ADB_DATA_DIR}/{model_id}")
    adb.push(str(model_path), f"{ADB_DATA_DIR}/{model_id}/model.dlc")

    command = ""
    if snpe_version == "2.32.6":
        command += "source /data/local/tmp/source_me.sh && "

    command += (
        "snpe-parallel-run "
        f"--container {ADB_DATA_DIR}/{model_id}/model.dlc "
        f"--input_list {ADB_DATA_DIR}/{dataset_id}/input_list.txt "
        f"--output_dir {ADB_DATA_DIR}/{model_id}/outputs "
        "--perf_profile default "
        "--cpu_fallback false "
        "--use_dsp"
    )
    ret, stdout, stderr = adb.shell(command)

    if ret != 0:
        raise SubprocessException(
            f"SNPE inference failed with code {ret}:\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}\n"
        )

    adb.pull(
        f"{ADB_DATA_DIR}/{model_id}/outputs",
        f"comparison/{model_id}/{snpe_version}/",
    )
    return Path("comparison", model_id, snpe_version)


def find_parent(instance: dict[str, Any]) -> dict[str, Any] | None:
    if instance["model_type"] == "ONNX":
        return instance
    parent_id = instance["parent_id"]
    if parent_id is None:
        return None

    return find_parent(request_info(parent_id, "modelInstances"))


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
        raise NotImplementedError("Multi stage archive")

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


def create_new_instance(
    instance_params: dict[str, Any], archive: Path
) -> None: ...


def migrate(
    old_instance: dict[str, Any],
    snpe_version: str,
    model_id: str,
    device_id: str | None = None,
) -> None:
    parent = find_parent(deepcopy(old_instance))
    if parent is None:
        logger.warning(
            f"Parent not found for model '{model_id}' and instance '{old_instance['id']}'"
        )
        return
    logger.info(
        f"Parent found for model '{model_id}' and instance '{old_instance['id']}': {parent['id']}"
    )
    precision = old_instance["model_precision_type"]

    old_archive = instance_download(
        old_instance["id"],
        output_dir=(MISC_DIR / "zoo" / old_instance["id"]),
        cache=True,
    )

    buildinfo_opts = get_buildinfo(old_archive)

    new_instance_params = get_instance_params(old_instance, parent)

    parent_archive = instance_download(
        parent["id"],
        output_dir=(MISC_DIR / "zoo" / parent["id"]),
        cache=True,
    )
    models_df["parent"].append(parent_archive.name)
    models_df["original"].append(old_archive.name)

    dataset_name = (
        df.filter(pl.col("Model ID") == model_id)
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

    if precision == "INT8":
        args.extend(
            [
                "calibration.path",
                CALIBRATION_DIR / "datasets" / dataset_name,
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
    new_dlc = next(iter((OUTPUTS_DIR / model_id).glob("*.dlc")))
    new_archive = next(iter((OUTPUTS_DIR / model_id).glob("*.tar.xz")))

    if test_degradation(
        old_archive, new_dlc, model_id, snpe_version, device_id
    ):
        logger.info(
            f"Degradation test passed for model '{model_id}' and instance '{old_instance['id']}'"
        )
        logger.info("Creating new instance")
        create_new_instance(new_instance_params, new_archive)
    else:
        logger.warning(
            f"Degradation test failed for model '{model_id}' and instance '{old_instance['id']}'"
        )


@app.default
def main(
    *,
    snpe_version: str = "2.32.6",
    dry: bool = True,
    device_id: str | None = None,
    model_id: str | None = None,
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
    """

    limit = 5 if dry else None
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

    for model in models:
        model_id = cast(str, model["id"])
        variants = _variant_ls(model_id=model_id, is_public=True, _silent=True)
        logger.info(f"Variants found: {len(variants)}")
        for variant in variants:
            if "RVC4" not in variant["platforms"]:
                continue

            instances = _instance_ls(
                model_id=model["id"],
                variant_id=variant["id"],
                model_type=None,
                is_public=True,
                _silent=True,
            )
            instances = get_missing_precision_instances(
                instances, snpe_version
            )
            logger.info(f"Instances found: {len(instances)}")
            for old_instance in instances:
                try:
                    migrate(old_instance, snpe_version, model_id, device_id)
                except Exception:
                    logger.exception(
                        f"Migration for model '{model_id}' failed!"
                    )

    pl.DataFrame(models_df).write_csv("downloaded_models.csv")


if __name__ == "__main__":
    app()
