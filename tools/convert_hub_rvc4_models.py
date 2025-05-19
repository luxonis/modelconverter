import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import polars as pl
from cyclopts import App
from loguru import logger
from luxonis_ml.data import LuxonisDataset, LuxonisLoader

from modelconverter.hub.__main__ import (
    _export,
    _instance_ls,
    _model_ls,
    _variant_ls,
    instance_delete,
)
from modelconverter.hub.__main__ import (
    instance_download as _instance_download,
)
from modelconverter.utils import AdbHandler
from modelconverter.utils.types import Target

instance_download = lru_cache(maxsize=None)(_instance_download)

app = App(name="convert_hub_rvc4_models")

adb = AdbHandler()

ADB_DATA_DIR = "/data/local/zoo_conversion/"
df = pl.read_csv("models.csv")


def get_missing_model_precisions(
    instance_list: list[dict[str, Any]], snpe_version: str
) -> set:
    model_precision_types = {
        inst["model_precision_type"]
        for inst in instance_list
        if inst["model_type"] == "RVC4"
    }
    snpe_version_model_precision_types = {
        inst["model_precision_type"]
        for inst in instance_list
        if inst["model_type"] == "RVC4"
        and inst["hardware_parameters"].get("snpe_version") == snpe_version
    }
    return model_precision_types - snpe_version_model_precision_types


def get_precision_to_params(
    model_id: str,
    instance_list: list[dict[str, Any]],
    precision_list: set[str],
    snpe_version: str,
    force_reexport: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        inst["model_precision_type"]: {
            "id": inst["id"],
            "parent_id": inst["parent_id"],
            "quantization_data": df.filter(pl.col("Model ID") == model_id)
            .select("Quant. Dataset ID")
            .item(),
            "snpe_version": inst["hardware_parameters"].get("snpe_version"),
            "input_shape": inst["input_shape"],
        }
        for inst in instance_list
        if inst["model_type"] == "RVC4"
        and (
            (
                inst["hardware_parameters"].get("snpe_version") != snpe_version
                and inst["model_precision_type"] in precision_list
            )
            or (
                force_reexport
                and inst["hardware_parameters"].get("snpe_version")
                == snpe_version
            )
        )
    }


def export_models(
    variant_info: dict[str, Any],
    target_precision: str,
    params: dict[str, Any],
    snpe_version: str,
    force_reexport: bool = False,
    **kwargs,
) -> Path:
    logger.info(
        f"Exporting: {variant_info['name']} {target_precision} SNPE {snpe_version}"
    )
    if force_reexport and params["snpe_version"] == snpe_version:
        logger.info(f"Force re-exporting: {params['id']}")
        instance_delete(params["id"])
    instance = _export(
        f"{variant_info['name']} {target_precision} SNPE {snpe_version}",
        params["parent_id"],
        Target.RVC4,
        target_precision=target_precision,
        quantization_data=params["quantization_data"],
        snpe_version=snpe_version,
        **kwargs,
    )
    return instance_download(instance["id"], output_dir="converted_models")


@lru_cache
def prepare_inference(dataset_id: str, width: int, height: int) -> None:
    dataset = LuxonisDataset(dataset_id, bucket_storage="gcs")
    loader = LuxonisLoader(
        dataset, width=width, height=height, keep_aspect_ratio=False
    )
    with tempfile.TemporaryDirectory() as d:
        input_list = ""
        for i, (img, _) in enumerate(loader):
            cv2.imwrite(f"{d}/{i}.jpg", img)
            input_list += f"{ADB_DATA_DIR}/{dataset_id}/{i}.jpg\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=d) as f:
            f.write(input_list)
            adb.push(f.name, f"{ADB_DATA_DIR}/{dataset_id}/input_list.txt")
        adb.push(d, f"{ADB_DATA_DIR}/{dataset_id}/")


def infer(
    model_path: Path,
    model_id: str,
    dataset_id: str,
    width: int,
    height: int,
    snpe_version: str,
) -> bool:
    prepare_inference(dataset_id, width, height)
    adb.shell(f"mkdir {ADB_DATA_DIR}/{model_id}")
    adb.push(str(model_path), f"{ADB_DATA_DIR}/{model_id}/model.dlc")
    ret, stdout, stderr = adb.shell(
        f"source /data/local/tmp/source_me_snpe_v{snpe_version}.sh && "
        "snpe-parallel-run "
        f"--container {ADB_DATA_DIR}/{model_id}/model.dlc "
        f"--input_list {ADB_DATA_DIR}/{dataset_id}/input_list.txt "
        f"--output_dir {ADB_DATA_DIR}/{model_id}/outputs "
        "--perf_profile default "
        "--cpu_fallback false "
        "--use_dsp"
    )
    if ret != 0:
        logger.error(f"Inference for model '{model_id}' failed!")
        logger.error(stdout)
        logger.error(stderr)
        return False
    adb.pull(
        f"{ADB_DATA_DIR}/{model_id}/outputs",
        f"comparison/{model_id}/{snpe_version}/",
    )
    return True


@app.default
def main(
    *,
    snpe_version: str = "2.32.6",
    force_reexport: bool = False,
    disable_onnx_simplification: bool = False,
    disable_onnx_optimization: bool = False,
    snpe_onnx_to_dlc_args: list[str] | None = None,
    snpe_dlc_quant_args: list[str] | None = None,
    snpe_dlc_graph_prepare_args: list[str] | None = None,
    limit: int = 3,
    is_public: bool = False,
) -> None:
    """Export all RVC4 models from the Luxonis Hub to SNPE format.

    Parameters
    ----------
    snpe_version : str
        The SNPE version to use for the export.
    force_reexport : bool
        If True, force re-export the model even if it already exists.
    disable_onnx_simplification : bool
        If True, disable ONNX simplification.
    disable_onnx_optimization : bool
        If True, disable ONNX optimization.
    snpe_onnx_to_dlc_args : list[str] | None
        Additional arguments to pass to the SNPE ONNX to DLC converter.
    snpe_dlc_quant_args : list[str] | None
        Additional arguments to pass to the SNPE DLC quantization tool.
    snpe_dlc_graph_prepare_args : list[str] | None
        Additional arguments to pass to the SNPE DLC graph preparation tool.
    """
    model_list = _model_ls(
        is_public=True,
        luxonis_only=True,
        limit=limit,
        _silent=True,
    )
    logger.info(f"Models found: {len(model_list)}")

    for model_info in model_list:
        model_id = model_info["id"]
        version_list = _variant_ls(
            model_id=model_id, is_public=is_public, _silent=True
        )

        for variant_info in version_list:
            if "RVC4" not in variant_info["platforms"]:
                continue

            instance_list = _instance_ls(
                model_id=model_info["id"],
                variant_id=variant_info["id"],
                model_type=None,
                is_public=True,
                _silent=True,
            )
            precision_list = get_missing_model_precisions(
                instance_list, snpe_version
            )
            logger.info(
                f"Missing model precisions for SNPE {snpe_version}: {precision_list}"
            )

            precision_to_params = get_precision_to_params(
                model_id,
                instance_list,
                precision_list,
                snpe_version,
                force_reexport,
            )
            for target_precision, params in precision_to_params.items():
                converted_dlc = export_models(
                    variant_info,
                    target_precision,
                    params,
                    snpe_version=snpe_version,
                    force_reexport=force_reexport,
                    disable_onnx_simplification=disable_onnx_simplification,
                    disable_onnx_optimization=disable_onnx_optimization,
                    snpe_onnx_to_dlc_args=snpe_onnx_to_dlc_args or [],
                    snpe_dlc_quant_args=snpe_dlc_quant_args or [],
                    snpe_dlc_graph_prepare_args=snpe_dlc_graph_prepare_args
                    or [],
                )
                orig_model = instance_download(params["parent_id"])

                adb.push(
                    str(converted_dlc), f"{ADB_DATA_DIR}/{model_id}/model.dlc"
                )


if __name__ == "__main__":
    app()
