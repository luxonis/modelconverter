import json
import math
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from os import getenv
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import onnxruntime as ort
import polars as pl
from cyclopts import App, Group
from data_manager import download_dataset, download_snpe_files
from loguru import logger
from luxonis_ml.nn_archive import Config
from metric import Metric

from modelconverter.cli.utils import get_configs
from modelconverter.hub.__main__ import (
    _instance_ls,
    instance_create,
    instance_download,
    upload,
)
from modelconverter.utils import AdbHandler, read_image, subprocess_run
from modelconverter.utils.config import (
    ImageCalibrationConfig,
    SingleStageConfig,
)
from modelconverter.utils.constants import (
    CALIBRATION_DIR,
    MISC_DIR,
    OUTPUTS_DIR,
    SHARED_DIR,
)
from modelconverter.utils.exceptions import SubprocessException
from modelconverter.utils.metadata import get_metadata
from modelconverter.utils.nn_archive import safe_members
from modelconverter.utils.types import DataType, Encoding, ResizeMethod

ADB_DATA_DIR = Path("/data/local/zoo_conversion/datasets")
ADB_MODELS_DIR = Path("/data/local/zoo_conversion/models")


@dataclass
class InputFileMapping:
    files_by_input: dict[str, dict[str, Path]]
    common_indices: set[str]


class SNPEMigrationTester:
    def __init__(
        self,
        snpe_target_version: str = "2.32.6",
        snpe_source_version: str = "2.23.0",
        mapping_csv: str = "zoo_migration/new_mappings.csv",
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
    ):
        self.infer_mode = infer_mode
        self.snpe_target_version = snpe_target_version
        self.snpe_source_version = snpe_source_version
        self.metric = metric
        self.limit = limit
        self.upload = upload
        self.confirm_upload = confirm_upload
        self.skip_conversion = skip_conversion
        self.verify = verify
        self.models = self._load_mapping(
            mapping_csv, model_id, variant_id, instance_id
        )
        self.results_df = {
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

        if infer_mode == "adb":
            self.adb_handler = AdbHandler(
                self._check_adb_connection(device_id)
            )
            self.infer = self._adb_infer

            old_version = self.snpe_source_version.replace(".", "_")
            snpe_path = MISC_DIR / ("snpe_" + old_version)
            download_snpe_files(old_version)
            self.adb_handler.push(
                snpe_path, f"/data/local/tmp/snpe_{old_version}"
            )
            self.adb_handler.shell(
                f"chmod -R 0755 /data/local/tmp/snpe_{old_version}"
            )

            new_version = self.snpe_target_version.replace(".", "_")
            snpe_path = MISC_DIR / ("snpe_" + new_version)
            download_snpe_files(new_version)
            self.adb_handler.push(
                snpe_path, f"/data/local/tmp/snpe_{new_version}"
            )
            self.adb_handler.shell(
                f"chmod -R 0755 /data/local/tmp/snpe_{new_version}"
            )

        elif infer_mode == "modelconv":
            self.infer = self._modelconv_infer
        else:
            raise ValueError(
                f"Invalid infer mode: {infer_mode}. Choose 'adb' or 'modelconv'."
            )

    def test_migration(self) -> None:
        """Main method to test the migration of models from one SNPE
        version to another."""
        logger.info(
            f"Testing migration from SNPE version {self.snpe_source_version} "
            f"to {self.snpe_target_version}. "
        )

        for row in self.models.iter_rows(named=True):
            model_id = row["model_id"]
            error = ""
            status = "passable"
            try:
                old_score, new_score, new_archive = self._migrate_single_model(
                    row
                )
            except (Exception, SubprocessException) as e:
                status = "failed"
                error = str(e)
                logger.exception(
                    f"Migration for model '{model_id}' failed with error: {e!s}"
                )

            if math.isclose(old_score, new_score, rel_tol=1e-3, abs_tol=1e-5):
                status = "success"
            self.results_df["model_name"].append(row["model_name"])
            self.results_df["model_id"].append(model_id)
            self.results_df["variant_id"].append(row["variant_id"])
            self.results_df["instance_id"].append(row["instance_id"])
            self.results_df["parent_id"].append(row["parent_id"])
            self.results_df["precision"].append(row["precision"])
            self.results_df["status"].append(status)
            self.results_df["error"].append(error)
            self.results_df["old_to_onnx_score"].append(old_score)
            self.results_df["new_to_onnx_score"].append(new_score)

            if self.upload:
                old_instance = self._load_instance(
                    row["variant_id"], row["instance_id"]
                )
                instance_params = self._get_instance_params(
                    row, old_instance, self.snpe_target_version
                )
                self.upload_new_instance(instance_params, new_archive)

        self._cleanup()

    def _migrate_single_model(
        self, row: dict[str, str]
    ) -> tuple[float, float, Path | None]:
        old_instance_id = row["instance_id"]
        model_id = row["model_id"]
        variant_id = row["variant_id"]
        parent_id = row["parent_id"]
        quant_dataset = row["quant_dataset"]
        test_dataset = row["test_dataset"]

        outdir = OUTPUTS_DIR / model_id / variant_id / f"{old_instance_id}_new"

        logger.info(
            f"Testing migration for model '{model_id}', "
            f"variant '{variant_id}', instance '{old_instance_id}'"
        )
        # Download models and data
        old_archive = instance_download(
            old_instance_id,
            output_dir=str(
                MISC_DIR / "zoo" / model_id / variant_id / old_instance_id
            ),
            cache=True,
        )

        parent_archive = instance_download(
            parent_id,
            output_dir=str(
                MISC_DIR / "zoo" / model_id / variant_id / parent_id
            ),
            cache=True,
        )

        test_dataset_path = download_dataset(dataset_id=test_dataset)

        buildinfo_opts, command_args = self.get_buildinfo(old_archive)
        precision = row["precision"]
        if precision is None:
            precision = self.predict_precision(command_args)
            logger.warning(
                f"Precision is `None`for model '{model_id}' "
                f"and instance '{old_instance_id}'. Predicted precision from buildinfo: {precision}"
            )

        if self.skip_conversion and outdir.exists():
            logger.info(
                f"Skipping conversion for model '{model_id}', variant {variant_id}, and instance '{old_instance_id}'"
            )
        else:  # convert to target SNPE version
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
                self.snpe_target_version,
                *buildinfo_opts,
            ]

            if precision == "INT8" and quant_dataset != "/":
                try:
                    quant_dataset_path = download_dataset(
                        dataset_id=quant_dataset
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to download quantization dataset '{quant_dataset}' for model '{model_id}' and instance '{old_instance_id}': {e}"
                    )
                    raise FileNotFoundError(
                        f"Quantization dataset '{quant_dataset}' not found for model '{model_id}' and instance '{old_instance_id}'."
                    ) from e
                args.extend(["calibration.path", quant_dataset_path])
                logger.info(
                    f"Using calibration dataset at {quant_dataset_path} for model '{model_id}' and instance '{old_instance_id}'"
                )
            else:
                args.extend(["rvc4.disable_calibration", "True"])

            logger.info(f"Running command: {' '.join(map(str, args))}")
            subprocess_run(args, silent=True)
            self.chown(SHARED_DIR)

        new_archive = next(iter((outdir).glob("*.tar.xz")))

        logger.info(
            f"Inferencing ONNX parent model {parent_id} for model '{model_id}' and instance '{old_instance_id}'"
        )
        onnx_outputs = self._onnx_infer(
            parent_archive, test_dataset_path, outdir
        )

        logger.info(
            f"Inferencing SNPE {self.snpe_source_version} results for model '{model_id}' and instance '{old_instance_id}'"
        )
        old_archive_outputs = self.infer(
            row,
            old_archive,
            test_dataset_path,
            self.snpe_source_version,
            outdir / str(self.snpe_source_version.replace(".", "_")),
        )

        logger.info(
            f"Inferencing SNPE {self.snpe_target_version} results for model '{model_id}' and instance '{old_instance_id}'"
        )
        new_archive_outputs = self.infer(
            row,
            new_archive,
            test_dataset_path,
            self.snpe_target_version,
            outdir / str(self.snpe_target_version.replace(".", "_")),
        )

        logger.info(
            f"Calculating metric {self.metric.value} for model '{model_id}' and instance '{old_instance_id}'"
        )

        old_score, new_score = self._calculate_metric(
            onnx_outputs, old_archive_outputs, new_archive_outputs
        )

        logger.info(
            f"New model {self.metric.value}: {new_score} {self.metric.sign} old model {self.metric.value}: {old_score}"
        )
        return old_score, new_score, new_archive

    def _onnx_infer(
        self,
        parent_archive: Path,
        test_dataset_path: Path,
        save_path: Path,
    ) -> Path:
        onnx_model_path, onnx_inputs, onnx_outputs = self.get_onnx_info(
            parent_archive, save_path
        )

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
        logger.info(f"Using dataset at '{test_dataset_path}'")

        outputs_path = test_dataset_path / "onnx" / "outputs"

        if outputs_path.exists():
            shutil.rmtree(outputs_path)
        outputs_path.mkdir(parents=True, exist_ok=True)

        mapping = self.validate_and_map_input_files(
            test_dataset_path, input_names
        )

        def run_onnx(
            session: ort.InferenceSession,
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
                files.extend(test_dataset_path.glob(pattern))

            if not files:
                raise RuntimeError(
                    f"No files found in dataset at {test_dataset_path}"
                )

            for file_path in sorted(files):
                name = input_names[0]
                shape = input_shapes[0]
                dtype = dtypes_map[input_dtypes[0]]
                prep = input_preprocessing[0]
                idx = file_path.stem.split("_")[-1]

                tensor = self.preprocess_image(file_path, shape, dtype, prep)
                run_onnx(session, onnx_outputs, {name: tensor})

        else:
            for idx in sorted(mapping.common_indices, key=lambda x: int(x)):
                input_files = {
                    name: mapping.files_by_input[name][idx]
                    for name in input_names
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
                        tensor = self.preprocess_image(
                            file_path, shape, dtypes_map[dtype], prep
                        )
                    input_tensors[name] = tensor
                run_onnx(session, onnx_outputs, input_tensors)

        return outputs_path

    def _adb_infer(
        self,
        model_info: dict,
        archive: Path,
        dataset: Path,
        snpe_version: str,
        save_path: Path,
    ) -> Path:
        if save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        model_name = model_info["model_name"]
        logger.info(
            f"Running inference on ADB. Parsing archive {model_name}..."
        )
        snpe_version_clean = "snpe_" + snpe_version.replace(".", "_")

        with (
            tempfile.TemporaryDirectory() as d,
            tarfile.open(archive, mode="r") as tf,
        ):
            tf.extractall(d, members=safe_members(tf))  # noqa: S202
            config = Config(**json.loads(Path(d, "config.json").read_text()))
            model_path = next(iter(Path(d).glob("*.dlc")))

            copied_model_path = save_path / "model.dlc"
            shutil.copy(model_path, copied_model_path)

            mult_cfg, _, _ = get_configs(str(archive))
            config = mult_cfg.get_stage_config(None)
            input_list_path, out_shapes = self.prepare_raw_files(
                dataset, config
            )

            # save_path =  OUTPUTS_DIR / model_id / variant_id / f"{old_instance_id}_new / snpe_version

            self.adb_handler.shell(f"mkdir -p {ADB_MODELS_DIR}")

            def source(snpe_version: str) -> str:
                return f"source /data/local/tmp/{snpe_version}/source_me.sh"

            self.adb_handler.push(model_path, ADB_MODELS_DIR / "model.dlc")

            subprocess_run(
                [
                    "modelconverter",
                    "shell",
                    "rvc4",
                    "--tool-version",
                    snpe_version,
                    "--command",
                    f"snpe-dlc-info -i {copied_model_path} -s {SHARED_DIR}/model_info.csv",
                ],
                silent=True,
            )

            metadata = get_metadata(SHARED_DIR / "model_info.csv")

            command = (
                f"{source(snpe_version_clean)} && "
                f"/data/local/tmp/{snpe_version_clean}/aarch64-oe-linux-gcc11.2/bin/snpe-net-run "
                f"--container {ADB_MODELS_DIR}/model.dlc "
                f"--input_list {input_list_path} "
                f"--output_dir {ADB_MODELS_DIR}/outputs "
                "--perf_profile balanced "
                "--use_dsp "
                "--userbuffer_floatN_output 32 "
                "--userbuffer_float "
            )

            ret, stdout, stderr = self.adb_handler.shell(command)

            if ret != 0:
                raise SubprocessException(
                    f"SNPE inference failed with code {ret}:\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}\n"
                )

            raw_out_dir = save_path / "raw"
            raw_out_dir.mkdir(parents=True, exist_ok=True)
            self.adb_handler.pull(f"{ADB_MODELS_DIR}/outputs", raw_out_dir)

            npy_out_dir = save_path / "npy"
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

    def _modelconv_infer(
        self,
        model_info: dict,
        archive: Path,
        dataset: Path,
        snpe_version: str,
        save_path: Path,
    ) -> Path:
        logger.info(
            f"Running inference on ModelConverter. Parsing archive {archive}..."
        )
        if save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        model_id: str = model_info["model_id"]
        instance_id: str = model_info["instance_id"]
        variant_id: str = model_info["variant_id"]
        dataset_id: str = str(dataset).split("/")[-1]

        with (
            tempfile.TemporaryDirectory() as d,
            tarfile.open(archive, mode="r") as tf,
        ):
            tf.extractall(d, members=safe_members(tf))  # noqa: S202
            config = Config(**json.loads(Path(d, "config.json").read_text()))
            model_path = next(iter(Path(d).glob("*.dlc")))
            inp_names = [inp.name for inp in config.model.inputs]
            new_model_path = save_path / "model.dlc"
            shutil.copy2(model_path, new_model_path)

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
                    CALIBRATION_DIR / "datasets" / dataset_id,
                    src,
                    dirs_exist_ok=True,
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
                new_model_path,
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

    def _calculate_metric(
        self,
        onnx_result_path: Path,
        old_snpe_result_path: Path,
        new_snpe_result_path: Path,
    ) -> tuple[float, float]:
        files = list(old_snpe_result_path.rglob("*.npy"))
        assert len(files) > 0, "No files found in old inference"

        scores_new_vs_onnx = []
        scores_old_vs_onnx = []

        for old_file in files:
            relative_path = old_file.relative_to(old_snpe_result_path)
            idx = int(old_file.stem.split("_")[-1])
            new_file = new_snpe_result_path / relative_path
            onnx_file = (
                onnx_result_path / relative_path
            ).parent / f"{idx}.npy"

            if not new_file.exists() or not onnx_file.exists():
                raise ValueError(
                    f"Some of the inferred files do not exists: {new_file}: {new_file.exists()}, {onnx_file}: {onnx_file.exists()}"
                )

            old_array = np.load(old_file)
            new_array = np.load(new_file)
            onnx_array = np.load(onnx_file)

            scores_new_vs_onnx.append(
                self.metric.compute(new_array, onnx_array)
            )
            scores_old_vs_onnx.append(
                self.metric.compute(old_array, onnx_array)
            )

        if not scores_new_vs_onnx or not scores_old_vs_onnx:
            raise RuntimeError(
                f"No scores computed for metric {self.metric.value}. "
            )

        old_score = float(np.mean(scores_old_vs_onnx))
        new_score = float(np.mean(scores_new_vs_onnx))

        if math.isnan(old_score) or math.isnan(new_score):
            raise RuntimeError(
                f"Degradation test failed: old model has NaN {self.metric.value} score ({old_score}) or new model has NaN {self.metric.value} score ({new_score})"
            )

        if math.isclose(old_score, new_score, rel_tol=5e-2, abs_tol=1e-5):
            return old_score, new_score

        self.metric.verify(old_score, new_score)
        return old_score, new_score

    def prepare_raw_files(
        self, dataset_path: Path, config: SingleStageConfig
    ) -> tuple[Path, dict[str, list[int] | None]]:
        input_names = [inp.name for inp in config.inputs]
        in_shapes = {inp.name: inp.shape for inp in config.inputs}
        out_shapes = {out.name: out.shape for out in config.outputs}

        resize_methods = {
            inp.name: inp.calibration.resize_method
            if isinstance(inp.calibration, ImageCalibrationConfig)
            else ResizeMethod.RESIZE
            for inp in config.inputs
        }
        encodings = {
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

        dataset_id = str(dataset_path).split("/")[-1]
        self.adb_handler.shell(f"mkdir -p {ADB_DATA_DIR}/{dataset_id}")

        mapping = self.validate_and_map_input_files(dataset_path, input_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            list_lines: list[str] = []

            if len(mapping.common_indices) == 0:
                file_patterns = ("*.jpg", "*.jpeg", "*.png", "*.npy")
                files: list[Path] = []
                for pattern in file_patterns:
                    files.extend(dataset_path.glob(pattern))

                if not files:
                    raise RuntimeError(
                        f"No files found in dataset {dataset_id}"
                    )

                for file_path in sorted(files):
                    name = input_names[0]
                    shape = in_shapes[name]
                    if shape is None:
                        logger.error(
                            f"Input shape for '{name}' is None. "
                            "Please provide a valid shape in the config."
                        )
                        raise ValueError
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
                for idx in sorted(
                    mapping.common_indices, key=lambda x: int(x)
                ):
                    input_files = {
                        name: mapping.files_by_input[name][idx]
                        for name in input_names
                    }

                    parts: list[str] = []

                    for name in input_names:
                        file_path = input_files[name]
                        in_shape = in_shapes[name]
                        if in_shape is None:
                            logger.error(
                                f"Input shape for '{name}' is None in the provided model config."
                            )
                            raise ValueError
                        n, *s, c = in_shape
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
            self.adb_handler.push(
                input_list_path, ADB_DATA_DIR / dataset_id / "input_list.txt"
            )
            input_list_path.unlink()
            self.adb_handler.push(tmpdir, ADB_DATA_DIR / dataset_id)

            return ADB_DATA_DIR / dataset_id / "input_list.txt", out_shapes

    def preprocess_image(
        self,
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
        scale = np.array(
            preprocessing.get("scale", scale_default), dtype=dtype
        )
        img = (img - mean) / scale

        return img.transpose(2, 0, 1)[None, ...]

    def validate_and_map_input_files(
        self, dataset_path: Path, input_names: list[str]
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
                ) from None

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

    def get_onnx_info(
        self, archive: Path, save_path: Path
    ) -> tuple[Path, list[dict[str, Any]], list[str]]:
        REMOVE_INP_KEYS = ("dtype", "input_type", "layout")
        REMOVE_PREP_KEYS = "interleaved_to_planar"

        def clean_input(inp: dict[str, Any]) -> dict[str, Any]:
            cleaned = {
                k: v for k, v in inp.items() if k not in REMOVE_INP_KEYS
            }
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

            input_name_to_meta = {
                inp.name: inp for inp in session.get_inputs()
            }

            updated_inputs = []
            for inp_dict in inputs:
                name = inp_dict.get("name")
                if name in input_name_to_meta:
                    onnx_input_meta = input_name_to_meta[name]
                    dtype_str = onnx_input_meta.type
                    if dtype_str.startswith("tensor(") and dtype_str.endswith(
                        ")"
                    ):
                        dtype = dtype_str[len("tensor(") : -1]
                    else:
                        logger.warning(
                            f"Unexpected ONNX input type format: {dtype_str} for input {name}"
                        )
                        dtype = "float32"  # Default to float32 if format is unexpected

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
            onnx_inputs = [
                clean_input(inp) for inp in config["model"]["inputs"]
            ]
            output_names = [
                outp["name"] for outp in config["model"]["outputs"]
            ]
            onnx_model = config["model"]["metadata"]["path"]
            tmp_onnx_path = next(iter(Path(d).glob(onnx_model)))
            if not tmp_onnx_path.exists():
                raise RuntimeError("ONNX file not found")
            onnx_inputs = get_input_dtype(tmp_onnx_path, onnx_inputs)

            self.chown(SHARED_DIR)
            dst = save_path / "onnx"
            dst.mkdir(parents=True, exist_ok=True)
            onnx_path = shutil.copy(tmp_onnx_path, dst / tmp_onnx_path.name)

        return Path(onnx_path), onnx_inputs, output_names

    def get_buildinfo(self, archive: Path) -> tuple[list[str], dict[str, Any]]:
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
            return (
                string.replace("'", "@@").replace('"', "'").replace("@@", '""')
            )

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

    def predict_precision(self, command_args: dict[str, Any]) -> str:
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

        return precision

    def _load_instance(
        self, variant_id: str, instance_id: str
    ) -> dict[str, Any]:
        """Load an instance from the Luxonis Hub."""
        try:
            instance = _instance_ls(
                model_version_id=variant_id,
                is_public=True,
                _silent=True,
            )

            instance = [inst for inst in instance if inst["id"] == instance_id]
            if len(instance) == 0:
                raise ValueError(
                    f"Instance {instance_id} not found in variant {variant_id}."
                )

            return cast(dict[str, Any], instance[0])
        except Exception as e:
            logger.error(
                f"Failed to load instance {instance_id} for variant {variant_id}: {e}"
            )
            raise FileNotFoundError(
                f"Instance {instance_id} not found in variant {variant_id}."
            ) from e

    def _get_instance_params(
        self,
        model_info: dict[str, Any],
        inst: dict[str, Any],
        snpe_version: str,
    ) -> dict[str, Any]:
        return {
            "name": inst["name"],
            "variant_id": model_info["variant_id"],
            "model_type": "RVC4",
            "parent_id": model_info["parent_id"],
            "hardware_parameters": {"snpe_version": snpe_version},
            "model_precision_type": inst["model_precision_type"],
            "quantization_data": model_info["quant_dataset"],
            "tags": inst["tags"],
            "input_shape": inst["input_shape"],
        }

    def upload_new_instance(
        self, instance_params: dict[str, Any], archive: Path | None
    ) -> None:
        logger.info("Creating new instance")
        instance = instance_create(**instance_params, silent=True)
        logger.info(
            f"New instance created: {instance['id']}, {instance['name']}"
        )
        upload(str(archive), instance["id"])

    def _check_adb_connection(self, device_id: str | None) -> str:
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
            return devices[0]
        if device_id not in devices:
            raise ValueError(
                f"Device ID '{device_id}' not found in connected devices: {devices}"
            )
        logger.info(f"Using device ID: {device_id}")

        return device_id

    def _load_mapping(
        self,
        mapping_csv: str,
        model_id: str | None = None,
        variant_id: str | None = None,
        instance_id: str | None = None,
    ) -> pl.DataFrame:
        """Load the mapping CSV file into a Polars DataFrame.

        Optionally filter by model_id, variant_id, and instance_id.
        """
        df = pl.read_csv(mapping_csv)

        if model_id is not None:
            df = df.filter(pl.col("model_id") == model_id)
        if variant_id is not None:
            df = df.filter(pl.col("variant_id") == variant_id)
        if instance_id is not None:
            df = df.filter(pl.col("instance_id") == instance_id)

        if df.is_empty():
            logger.error(
                "The desired model is not part of the public zoo model list."
            )

        return df

    def _cleanup(self) -> None:
        date = datetime.now().strftime("%Y_%m_%d_%H_%M")  # noqa: DTZ005

        pl_df = pl.DataFrame(self.results_df)
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

    def chown(self, path: Path) -> None:
        subprocess_run(f"sudo chown -R {getenv('USER')} {path}", silent=True)


app = App(name="zoo_migration", help="Zoo migration commands")


@app.meta.command(group=Group.create_ordered("Migration Commands"))
def test(
    snpe_target_version: str = "2.32.6",
    snpe_source_version: str = "2.23.0",
    mapping_csv: str = "zoo_migration/mapping.csv",
    device_id: str | None = None,
    model_id: str | None = None,
    variant_id: str | None = None,
    instance_id: str | None = None,
    infer_mode: Literal["adb", "modelconv"] = "adb",
    metric: Metric = Metric.MSE,
    limit: int = 5,
    upload: bool = False,
    confirm_upload: bool = False,
    skip_conversion: bool = False,
    verify: bool = True,
) -> None:
    tester = SNPEMigrationTester(
        snpe_target_version=snpe_target_version,
        snpe_source_version=snpe_source_version,
        mapping_csv=mapping_csv,
        device_id=device_id,
        model_id=model_id,
        variant_id=variant_id,
        instance_id=instance_id,
        infer_mode=infer_mode,
        metric=metric,
        limit=limit,
        upload=upload,
        confirm_upload=confirm_upload,
        skip_conversion=skip_conversion,
        verify=verify,
    )
    tester.test_migration()


if __name__ == "__main__":
    app.meta()
