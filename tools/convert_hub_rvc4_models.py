import argparse
from typing import Any

from loguru import logger

from modelconverter.hub.__main__ import (
    _export,
    instance_delete,
    instance_ls,
    model_ls,
    variant_ls,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert models from Luxonis Hub to RVC4 format using the specified SNPE version. The script requires HUBAI_API_KEY to be set in the environment variables."
    )
    parser.add_argument(
        "--snpe_version",
        type=str,
        default="2.23.0",
        help="SNPE version to use for conversion.",
    )
    parser.add_argument(
        "--force-reexport",
        action="store_true",
        help="Force re-export of models.",
    )
    parser.add_argument(
        "--disable_onnx_simplification",
        action="store_true",
        help="Disable ONNX simplification.",
    )
    parser.add_argument(
        "--disable_onnx_optimization",
        action="store_true",
        help="Disable ONNX optimization.",
    )
    parser.add_argument(
        "--snpe_onnx_to_dlc_args",
        type=str,
        nargs="+",
        default=None,
        help="SNPE ONNX to DLC arguments.",
    )
    parser.add_argument(
        "--snpe_dlc_quant_args",
        type=str,
        nargs="+",
        default=None,
        help="SNPE DLC quantization arguments.",
    )
    parser.add_argument(
        "--snpe_dlc_graph_prepare_args",
        type=str,
        nargs="+",
        default=None,
        help="SNPE DLC graph prepare arguments.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Limit the number of models to process.",
    )
    parser.add_argument(
        "--is_public", action="store_true", help="Process public models."
    )
    return parser.parse_args()


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
    instance_list: list[dict[str, Any]],
    precision_list: set[str],
    snpe_version: str,
    force_reexport: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        inst["model_precision_type"]: {
            "id": inst["id"],
            "parent_id": inst["parent_id"],
            "quantization_data": inst["quantization_data"],
            "snpe_version": inst["hardware_parameters"].get("snpe_version"),
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
    precision_to_params: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    target_options = {
        "snpe_version": args.snpe_version,
        "disable_onnx_simplification": args.disable_onnx_simplification,
        "disable_onnx_optimization": args.disable_onnx_optimization,
        "snpe_onnx_to_dlc_args": args.snpe_onnx_to_dlc_args,
        "snpe_dlc_quant_args": args.snpe_dlc_quant_args,
        "snpe_dlc_graph_prepare_args": args.snpe_dlc_graph_prepare_args,
    }
    target_options = {k: v for k, v in target_options.items() if v is not None}

    for target_precision, params in precision_to_params.items():
        logger.info(
            f"Exporting: {variant_info['name']} {target_precision} SNPE {args.snpe_version}"
        )
        if args.force_reexport and params["snpe_version"] == args.snpe_version:
            logger.info(f"Force re-exporting: {params['id']}")
            instance_delete(params["id"])
        _export(
            f"{variant_info['name']} {target_precision} SNPE {args.snpe_version}",
            params["parent_id"],
            "rvc4",
            target_precision=target_precision,
            quantization_data=params["quantization_data"],
            **target_options,
        )


def main() -> None:
    args = parse_arguments()
    model_list = model_ls(
        is_public=args.is_public, luxonis_only=True, limit=args.limit
    )
    logger.info(f"Models found: {len(model_list)}")

    for model_info in model_list:
        version_list = variant_ls(
            model_id=model_info["id"], is_public=args.is_public
        )
        logger.info(f"Variants for {model_info['id']}: {version_list}")

        for variant_info in version_list:
            if "RVC4" not in variant_info["platforms"]:
                continue

            instance_list = instance_ls(
                model_id=model_info["id"],
                variant_id=variant_info["id"],
                model_type=None,
                is_public=args.is_public,
            )
            precision_list = get_missing_model_precisions(
                instance_list, args.snpe_version
            )
            logger.info(
                f"Missing model precisions for SNPE {args.snpe_version}: {precision_list}"
            )

            precision_to_params = get_precision_to_params(
                instance_list,
                precision_list,
                args.snpe_version,
                args.force_reexport,
            )
            export_models(variant_info, precision_to_params, args)


if __name__ == "__main__":
    main()
