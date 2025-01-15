import json
import os
import shutil
from pathlib import Path
from typing import Tuple

import wget
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import InputType

from modelconverter.cli import Request
from modelconverter.utils import ONNXModifier, environ
from modelconverter.utils.config import Config
from modelconverter.utils.onnx_tools import onnx_attach_normalization_to_inputs

DATA_DIR = Path("tests/data/test_utils/hub_ai_models")

HEADERS = {"Authorization": f"Bearer {environ.HUBAI_API_KEY}"}

EXCEMPTED_MODELS = [
    "l2cs",
    "zero-dce-400x600",
    "mult_640x352",
    "mult_512x288",
]

EXCEMPT_OPTIMISATION = [
    "efficientvit-b1-224",
]


def download_onnx_models():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    hub_ai_models = Request.get(
        "models/", params={"is_public": True, "limit": 1000}
    )

    for model in hub_ai_models:
        if "ONNX" in model["exportable_types"]:
            model_name = model["name"]
            model_dir = DATA_DIR / f"{model_name}"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_id = model["id"]

            model_variants = Request.get(
                "modelVersions/",
                params={
                    "model_id": model_id,
                    "is_public": True,
                    "limit": 1000,
                },
            )

            for variant in model_variants:
                if "ONNX" in variant["exportable_types"]:
                    model_version_id = variant["id"]
                    break
            download_info = Request.get(
                f"modelVersions/{model_version_id}/download"
            )

            model_download_link = download_info[0]["download_link"]

            filename = wget.download(
                model_download_link, out=model_dir.as_posix()
            )

            if filename.endswith(".tar.xz"):
                shutil.unpack_archive(filename, model_dir.as_posix())

                with open(model_dir / "config.json") as f:
                    cfg = json.load(f)
                model_name = cfg["model"]["metadata"]["path"].split(".onnx")[0]

                shutil.move(filename, model_dir / f"{model_name}.tar.xz")
                shutil.move(
                    model_dir / "config.json",
                    model_dir / f"{model_name}_config.json",
                )

                for item in Path(model_dir).iterdir():
                    shutil.move(str(item), DATA_DIR / item.name)

                shutil.rmtree(model_dir)
            else:
                os.remove(filename)

    onnx_models = []
    for onnx_file in DATA_DIR.glob("*.onnx"):
        if (
            onnx_file.stem not in EXCEMPTED_MODELS
            and "_modified" not in onnx_file.stem
        ):
            onnx_models.append(onnx_file)
    return onnx_models


def get_config(nn_config: Path) -> Tuple[Config, str]:
    with open(nn_config) as f:
        archive_config = NNArchiveConfig(**json.load(f))

    main_stage_config = {
        "input_model": str(DATA_DIR / archive_config.model.metadata.path),
        "inputs": [],
        "outputs": [],
    }

    for inp in archive_config.model.inputs:
        reverse = inp.preprocessing.reverse_channels
        interleaved_to_planar = inp.preprocessing.interleaved_to_planar
        dai_type = inp.preprocessing.dai_type

        layout = inp.layout
        encoding = "NONE"
        if inp.input_type == InputType.IMAGE:
            if dai_type is not None:
                if dai_type.startswith("RGB"):
                    encoding = {"from": "RGB", "to": "BGR"}
                elif dai_type.startswith("BGR"):
                    encoding = "BGR"
                elif dai_type.startswith("GRAY"):
                    encoding = "GRAY"
                else:
                    encoding = {"from": "RGB", "to": "BGR"}

                if dai_type.endswith("i"):
                    layout = "NHWC"
                elif dai_type.endswith("p"):
                    layout = "NCHW"
            else:
                if reverse is not None:
                    if reverse:
                        encoding = {"from": "RGB", "to": "BGR"}
                    else:
                        encoding = "BGR"
                else:
                    encoding = {"from": "RGB", "to": "BGR"}

                if interleaved_to_planar is not None:
                    if interleaved_to_planar:
                        layout = "NHWC"
                    else:
                        layout = "NCHW"
            channels = (
                inp.shape[layout.index("C")]
                if layout and "C" in layout
                else None
            )
            if channels and channels == 1:
                encoding = "GRAY"

        mean = inp.preprocessing.mean or [0, 0, 0]
        scale = inp.preprocessing.scale or [1, 1, 1]

        main_stage_config["inputs"].append(
            {
                "name": inp.name,
                "shape": inp.shape,
                "layout": layout,
                "data_type": inp.dtype.value,
                "mean_values": mean,
                "scale_values": scale,
                "encoding": encoding,
            }
        )

    for out in archive_config.model.outputs:
        main_stage_config["outputs"].append(
            {
                "name": out.name,
                "shape": out.shape,
                "layout": out.layout,
                "data_type": out.dtype.value,
            }
        )

    main_stage_key = archive_config.model.metadata.name
    config = {
        "name": main_stage_key,
        "stages": {
            main_stage_key: main_stage_config,
        },
    }

    for head in archive_config.model.heads or []:
        postprocessor_path = getattr(head.metadata, "postprocessor_path", None)
        if postprocessor_path is not None:
            input_model_path = DATA_DIR / postprocessor_path
            head_stage_config = {
                "input_model": str(input_model_path),
                "inputs": [],
                "outputs": [],
                "encoding": "NONE",
            }
            config["stages"][input_model_path.stem] = head_stage_config

    return Config.get_config(config, None), main_stage_key


def pytest_generate_tests(metafunc):
    params = download_onnx_models()
    metafunc.parametrize("onnx_file", params)


def test_onnx_model(onnx_file):
    skip_optimisation = (
        True if onnx_file.stem in EXCEMPT_OPTIMISATION else False
    )
    nn_config = onnx_file.parent / f"{onnx_file.stem}_config.json"
    cfg, main_stage_key = get_config(nn_config)

    input_configs = {
        input_config.name: input_config
        for input_config in cfg.stages[main_stage_key].inputs
    }
    for input_name in input_configs:
        input_configs[input_name].layout = "NCHW"

    modified_onnx = onnx_file.parent / f"{onnx_file.stem}_modified.onnx"
    onnx_attach_normalization_to_inputs(
        onnx_file, modified_onnx, input_configs
    )

    modified_optimised_onnx = (
        onnx_file.parent / f"{onnx_file.stem}_modified_optimised.onnx"
    )
    onnx_modifier = ONNXModifier(
        model_path=modified_onnx,
        output_path=modified_optimised_onnx,
        skip_optimisation=skip_optimisation,
    )

    if onnx_modifier.has_dynamic_shape:
        return

    assert (
        onnx_modifier.modify_onnx() and onnx_modifier.compare_outputs()
    ), f"Test failed for {onnx_file.name}"
