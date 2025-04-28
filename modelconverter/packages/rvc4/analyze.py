import os
import shutil
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnx.onnx_pb
import onnxruntime as rt
import polars as pl
from PIL import Image

from modelconverter.packages.base_analyze import Analyzer
from modelconverter.utils import AdbHandler, constants, subprocess_run


class RVC4Analyzer(Analyzer):
    def __init__(self, dlc_model_path: str, image_dirs: dict[str, str]):
        super().__init__(dlc_model_path, image_dirs)
        self.adb = AdbHandler()

    def analyze_layer_outputs(self, onnx_model_path: Path) -> None:
        input_matcher = self._prepare_input_matcher()
        dlc_matcher = self._prepare_raw_inputs(input_matcher, np.float32)

        output_dir = Path(
            self._run_dlc(
                f"snpe-net-run --container {self.model_name}.dlc --input_list input_list.txt --debug --use_dsp --userbuffer_floatN_output 32 --perf_profile balanced --userbuffer_float"
            )
        )
        dlc_matcher = {k: output_dir / v for k, v in dlc_matcher.items()}

        self._flatten_dlc_outputs(dlc_matcher)
        self._compare_to_onnx(str(onnx_model_path), input_matcher, dlc_matcher)

        self._cleanup_dlc_outputs()

    def _resize_image(
        self, img_path: str, input_sizes: list[int]
    ) -> np.ndarray:
        image = Image.open(img_path)
        image = image.resize(input_sizes)
        image = np.array(image)
        image = image[:, :, ::-1]

        return image.astype(np.uint8)

    def _prepare_input_matcher(self) -> dict[str, dict[str, str]]:
        image_names = {
            k: sorted(Path(v).glob("*")) for k, v in self.image_dirs.items()
        }
        if len({len(v) for v in image_names.values()}) != 1:
            raise ValueError(
                "All input dirs must have the same number of input images"
            )

        input_matcher = {}
        for i in range(len(next(iter(image_names.values())))):
            input_matcher[i] = {}
            for k in image_names:
                input_matcher[i][k] = str(image_names[k][i])

        return input_matcher

    def _prepare_raw_inputs(
        self, input_matcher: dict[str, dict[str, str]], type: type = np.uint8
    ) -> dict[str, str]:
        self.adb.shell(f"rm -rf /data/local/tmp/{self.model_name}")
        self.adb.shell(f"mkdir -p /data/local/tmp/{self.model_name}/inputs")

        input_list = ""
        dlc_matcher = {}
        for i, input_dict in input_matcher.items():
            input_row = ""
            dlc_matcher[i] = "Result_" + str(i)
            for input_name, img_path in input_dict.items():
                if not img_path.endswith((".png", ".jpg")):
                    continue
                img_name = Path(img_path).name
                width_height = self.input_sizes[input_name][1:3][::-1]
                image = self._resize_image(img_path, width_height)
                image = image.astype(type)
                raw_image = cast(np.ndarray, image).astype(type)

                with tempfile.NamedTemporaryFile() as f:
                    raw_image.tofile(f)
                    self.adb.push(
                        f.name,
                        f"/data/local/tmp/{self.model_name}/inputs/{img_name}.raw",
                    )
                    f.close()

                input_row += f"{input_name}:=/data/local/tmp/{self.model_name}/inputs/{img_name}.raw"
            input_list += input_row
            input_list += "\n"

        with tempfile.NamedTemporaryFile() as f:
            f.write(input_list.encode())
            f.flush()
            self.adb.push(
                f.name, f"/data/local/tmp/{self.model_name}/input_list.txt"
            )
            f.close()

        return dlc_matcher

    def _add_outputs_to_all_layers(self, onnx_file_path: str) -> Path:
        if Path(onnx_file_path.replace(".onnx", "-all-layers.onnx")).exists():
            Path(onnx_file_path.replace(".onnx", "-all-layers.onnx")).unlink()

        model = onnx.load(onnx_file_path)
        onnx.checker.check_model(model)
        graph = model.graph
        orig_graph_output = list(graph.output)

        del graph.output[:]

        for node in graph.node:
            for output in node.output:
                if output not in [o.name for o in orig_graph_output]:
                    value_info = onnx.onnx_pb.ValueInfoProto()
                    value_info.name = output
                    graph.output.append(value_info)

        for output in orig_graph_output:
            graph.output.append(output)
        all_output_name = onnx_file_path.replace(".onnx", "-all-layers.onnx")
        onnx.save(model, all_output_name)

        return Path(all_output_name)

    def _compare_to_onnx(
        self,
        onnx_model_path: str,
        input_matcher: dict[str, dict[str, str]],
        dlc_matcher: dict[str, Path],
    ) -> None:
        onnx_all_layers = self._add_outputs_to_all_layers(str(onnx_model_path))
        session = rt.InferenceSession(onnx_all_layers)
        output_names = [layer.name for layer in session.get_outputs()]

        layer_names = self._replace_bad_layer_names(output_names)

        statistics = []
        onnx_input_shapes = {}
        for input_metadata in session.get_inputs():
            onnx_input_shapes[input_metadata.name] = input_metadata.shape

        for i, input_dict in input_matcher.items():
            onnx_input_dict = {}
            for input_name, img_path in input_dict.items():
                if not img_path.endswith((".png", ".jpg")):
                    continue

                shape = onnx_input_shapes[input_name][2:][::-1]
                image = self._resize_image(img_path, shape)
                image = np.transpose(
                    image, [2, 0, 1]
                )  # NCHW format is assumed by default and resize returns HWC
                image = np.expand_dims(image, axis=0).astype(np.float32)

                onnx_input_dict[input_name] = image

            outputs = session.run(output_names, onnx_input_dict)

            dlc_output_path = dlc_matcher[i]

            for layer_name, onnx_layer_output in zip(
                layer_names, outputs, strict=True
            ):
                dlc_layer_size = self.output_sizes.get(layer_name)
                if dlc_layer_size is None:
                    continue

                with open(dlc_output_path / f"{layer_name}.raw", "rb") as f:
                    raw_data = f.read()

                dlc_layer_output = np.frombuffer(raw_data, dtype=np.float32)
                dlc_layer_output = dlc_layer_output.reshape(dlc_layer_size)

                if dlc_layer_output.shape != onnx_layer_output.shape:
                    dlc_layer_output = np.transpose(
                        dlc_layer_output, [0, 3, 1, 2]
                    )

                layer_stats = self._calculate_statistics(
                    onnx_layer_output, dlc_layer_output
                )
                statistics.append([layer_name, *layer_stats])

        output_dir = f"{constants.OUTPUTS_DIR!s}/analysis/{self.model_name}"
        stats_df = pl.DataFrame(
            statistics,
            schema=["layer_name", "max_abs_diff", "MSE", "cos_sim"],
            orient="row",
        )
        grouped_df = stats_df.group_by("layer_name").agg(
            [
                pl.col("max_abs_diff").mean().alias("max_abs_diff"),
                pl.col("MSE").mean().alias("MSE"),
                pl.col("cos_sim").mean().alias("cos_sim"),
            ]
        )

        layer_mapping = {name: idx for idx, name in enumerate(layer_names)}
        grouped_df = grouped_df.with_columns(
            pl.col("layer_name")
            .map_elements(
                lambda x: layer_mapping.get(x, -1), return_dtype=pl.Int32
            )
            .alias("order")
        ).sort("order")

        grouped_df = grouped_df.drop("order")
        grouped_df.write_csv(f"{output_dir}/layer_comparison.csv")
        Path(onnx_all_layers).unlink()

    def _calculate_statistics(
        self, onnx_output: np.ndarray, dlc_output: np.ndarray
    ) -> list:
        max_abs_diff = np.max(np.abs(onnx_output - dlc_output))
        mse = np.mean((onnx_output - dlc_output) ** 2)
        cosine_sim = np.dot(onnx_output.flatten(), dlc_output.flatten()) / (
            np.linalg.norm(onnx_output.flatten())
            * np.linalg.norm(dlc_output.flatten())
        )

        return [max_abs_diff, mse, cosine_sim]

    def _run_dlc(self, command: str) -> str:
        self.adb.push(
            str(self.dlc_model_path),
            f"/data/local/tmp/{self.model_name}/{self.model_name}.dlc",
        )
        self.adb.shell(f"rm -rf /data/local/tmp/{self.model_name}/output")
        self.adb.shell(f"cd /data/local/tmp/{self.model_name} && {command}")

        target_dir = constants.OUTPUTS_DIR / "analysis" / self.model_name
        if (target_dir / "output").exists():
            shutil.rmtree(target_dir / "output")

        target_dir.mkdir(parents=True, exist_ok=True)
        self.adb.pull(
            f"/data/local/tmp/{self.model_name}/output", f"{target_dir}/output"
        )

        return f"{target_dir}/output"

    def _flatten_dlc_outputs(self, dlc_matcher: dict[str, Path]) -> None:
        for result_path in dlc_matcher.values():
            # TODO: replace with `iterdir`
            for root, _, files in os.walk(result_path):
                for file in files:
                    relative_path = os.path.relpath(root, result_path)
                    new_file_name = (
                        relative_path.replace(os.sep, "_") + f"_{file}"
                    )
                    new_file_name = new_file_name.strip(".raw")
                    new_file_name = self._replace_bad_layer_names(
                        [new_file_name]
                    )[0]

                    source_path = str(Path(root, file))
                    if source_path not in f"{result_path}/{new_file_name}.raw":
                        shutil.copy(
                            source_path, f"{result_path}/{new_file_name}.raw"
                        )

            with os.scandir(result_path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        shutil.rmtree(entry.path)

    # layer execution times
    def analyze_layer_cycles(self) -> None:
        input_matcher = self._prepare_input_matcher()
        _ = self._prepare_raw_inputs(input_matcher)

        output_dir = self._run_dlc(
            f"snpe-net-run --container {self.model_name}.dlc --input_list input_list.txt --use_dsp --use_native_input_files --use_native_output_files --perf_profile balanced --userbuffer_tf8"
        )

        csv_path = Path(output_dir + "/layer_stats.csv")
        subprocess_run(
            [
                "snpe-diagview",
                "--input_log",
                output_dir + "/SNPEDiag.log",
                "--csv_format_version",
                "2",
                "--output",
                csv_path,
            ],
            silent=True,
        )
        self._process_diagview_csv(str(csv_path))
        csv_path.unlink()

        shutil.rmtree(output_dir)
        self._cleanup_dlc_outputs()

    def _process_diagview_csv(self, csv_path: str) -> None:
        df = pl.read_csv(csv_path)
        df = df.drop_nans()
        df = df.drop_nulls()
        layer_stats = df.group_by("Layer Id").agg(
            pl.col("Time").mean().round(0).cast(int).alias("time_mean"),
            pl.col("Layer Name").first().alias("layer_name"),
            pl.col("Unit of Measurement").first().alias("unit"),
        )
        layer_stats = layer_stats.rename({"Layer Id": "layer_id"})

        total_time = layer_stats["time_mean"].sum()
        layer_stats = layer_stats.with_columns(
            pl.col("layer_id").cast(int).alias("layer_id"),
            pl.col("layer_name")
            .str.split(":")
            .list.first()
            .map_elements(
                lambda x: self._replace_bad_layer_name(x), return_dtype=pl.Utf8
            )
            .alias("layer_name"),
            pl.col("time_mean")
            .mul(1 / total_time)
            .alias("Percentage_of_Total_Time"),
        )

        layer_stats = layer_stats.sort("layer_id")

        layer_stats.write_csv(
            f"{constants.OUTPUTS_DIR!s}/analysis/{self.model_name}/layer_cycles.csv",
        )

    # cleanup
    def _cleanup_dlc_outputs(self) -> None:
        self.adb.shell(f"rm -rf /data/local/tmp/{self.model_name}")

        output_dir = Path(
            f"{constants.OUTPUTS_DIR!s}/analysis/{self.model_name}/output"
        )
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
