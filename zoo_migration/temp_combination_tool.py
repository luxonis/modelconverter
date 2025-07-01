from copy import deepcopy
from datetime import datetime
from typing import Any, cast

import pandas as pd

from modelconverter.cli.utils import request_info
from modelconverter.hub.__main__ import (
    _instance_ls,
    _model_ls,
    _variant_ls,
)

mappings = pd.read_csv("zoo_migration/mappings_old.csv")


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


def find_parent(instance: dict[str, Any]) -> dict[str, Any] | None:
    if instance["model_type"] == "ONNX":
        return instance
    parent_id = instance["parent_id"]
    if parent_id is None:
        return None

    return find_parent(request_info(parent_id, "modelInstances"))


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


models = _model_ls(
    is_public=True,
    luxonis_only=True,
    _silent=False,
)

new_mappings = {
    "model_name": [],
    "model_id": [],
    "variant_id": [],
    "instance_id": [],
    "parent_id": [],
    "precision": [],
    "quant_dataset": [],
    "test_dataset": [],
}

for model in models:
    model_id = cast(str, model["id"])
    if model_id == "6684e96f-11fc-4d92-8657-12a5fd8e532a":
        continue  # 6684e96f is YOLO-World L that has no uploaded parent or quant / test dataset

    variants = _variant_ls(model_id=model_id, is_public=True, _silent=True)
    variants = [v for v in variants if "RVC4" in v["platforms"]]
    for variant in variants:
        version_id = cast(str, variant["id"])

        all_instances = _instance_ls(
            model_version_id=version_id,
            model_type=None,
            is_public=True,
            _silent=True,
        )

        row = mappings[(mappings["Model ID"] == model_id)]

        if not any(pd.isna(row["Variant ID"])):
            row = row[(row["Variant ID"] == version_id)]
        if row.empty:
            continue

        instances = [
            instance
            for instance in all_instances
            if "RVC4" in instance["platforms"]
        ]
        instances = get_missing_precision_instances(all_instances, "2.23.0")

        for instance in instances:
            parent = find_parent(deepcopy(instance))
            if parent is None:
                parent = guess_parent(instance, all_instances)

            new_mappings["model_name"].append(model["name"])
            new_mappings["model_id"].append(model_id)
            new_mappings["variant_id"].append(version_id)
            new_mappings["instance_id"].append(instance["id"])
            new_mappings["parent_id"].append(parent["id"] if parent else None)
            new_mappings["precision"].append(instance["model_precision_type"])
            quant_dataset_name = row["Quant. Dataset ID"].values[0]
            test_dataset_name = row["Test Dataset ID"].values[0]
            if quant_dataset_name.endswith("Dataset from Hub"):
                quant_dataset_name = f"{quant_dataset_name.split()[0].lower()}_quantization_data"

            new_mappings["quant_dataset"].append(quant_dataset_name)
            new_mappings["test_dataset"].append(test_dataset_name)

df = pd.DataFrame(new_mappings)

df.to_csv("zoo_migration/new_mappings.csv", index=False)
