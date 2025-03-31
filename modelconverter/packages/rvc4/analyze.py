import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import onnx
import onnx.onnx_pb
import onnxruntime as rt
import pandas as pd
from PIL import Image

from modelconverter.utils import AdbHandler, constants, subprocess_run

from ..base_analyze import Analyzer


class RVC4Analyzer(Analyzer):
    def __init__(self, dlc_model_path: str, image_dirs: Dict[str, str]):
        super().__init__(dlc_model_path, image_dirs)
        self.adb = AdbHandler()

    def analyze_layer_outputs(self, onnx_model_path: Path) -> None:
        input_matcher = self._prepare_input_matcher()
        dlc_matcher = self._prepare_raw_inputs(input_matcher, np.float32)

        output_dir = self._run_dlc(
            f"snpe-net-run --container {self.model_name}.dlc --input_list input_list.txt --debug --use_dsp --userbuffer_floatN_output 32 --perf_profile balanced --userbuffer_float"
        )
        dlc_matcher = {
            k: os.path.join(output_dir, v) for k, v in dlc_matcher.items()
        }

        self._flatten_dlc_outputs(dlc_matcher)
        self._compare_to_onnx(str(onnx_model_path), input_matcher, dlc_matcher)

        self._cleanup_dlc_outputs()

    def _resize_image(
        self, img_path: str, input_sizes: List[int]
    ) -> np.ndarray:
        image = Image.open(img_path)
        image = image.resize(input_sizes)
        image = np.array(image)
        image = image[:, :, ::-1]

        return image.astype(np.uint8)

    def _prepare_input_matcher(self) -> Dict[str, Dict[str, str]]:
        image_names = {
            k: sorted(Path(v).glob("*")) for k, v in self.image_dirs.items()
        }
        if len(set([len(v) for v in image_names.values()])) != 1:
            raise ValueError(
                "All input dirs must have the same number of input images"
            )

        input_matcher = {}
        for i in range(len(next(iter(image_names.values())))):
            input_matcher[i] = {}
            for k in image_names.keys():
                input_matcher[i][k] = str(image_names[k][i])

        # input_matcher = { 0 : {input_name1 : full_path_to_img1, input_name2 : full_path_to_img2},
        #                   1 : {input_name1 : full_path_to_img2, input_name2 : full_path_to_img2},
        #                   ....
        #                   n : {input_name1 : full_path_to_imgn, input_name2 : full_path_to_imgn}}
        return input_matcher

    def _prepare_raw_inputs(
        self, input_matcher: Dict[str, Dict[str, str]], type: type = np.uint8
    ) -> Dict[str, str]:
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
                img_name = os.path.splitext(os.path.basename(img_path))[0]
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

    def _add_outputs_to_all_layers(self, onnx_file_path: str) -> str:
        if os.path.exists(onnx_file_path.replace(".onnx", "-all-layers.onnx")):
            os.remove(
                Path(onnx_file_path.replace(".onnx", "-all-layers.onnx"))
            )

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

        return all_output_name

    def _compare_to_onnx(
        self,
        onnx_model_path: str,
        input_matcher: Dict[str, Dict[str, str]],
        dlc_matcher: Dict[str, str],
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

            for layer_name, onnx_layer_output in zip(layer_names, outputs):
                dlc_layer_size = self.output_sizes.get(layer_name)
                if dlc_layer_size is None:
                    continue

                with open(
                    os.path.join(dlc_output_path, f"{layer_name}.raw"), "rb"
                ) as f:
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

        output_dir = f"{str(constants.OUTPUTS_DIR)}/analysis/{self.model_name}"
        stats_df = pd.DataFrame(
            statistics,
            columns=["layer_name", "max_abs_diff", "MSE", "cos_sim"],
        )
        stats_df = stats_df.groupby("layer_name").agg("mean").reset_index()

        stats_df["layer_name"] = pd.Categorical(
            stats_df["layer_name"], categories=layer_names, ordered=True
        )
        stats_df = stats_df.sort_values("layer_name")
        stats_df.to_csv(f"{output_dir}/layer_comparison.csv", index=False)

        os.remove(Path(onnx_all_layers))

    def _calculate_statistics(
        self, onnx_output: np.ndarray, dlc_output: np.ndarray
    ) -> List:
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

        target_dir = (
            f"{str(constants.OUTPUTS_DIR)}/analysis/{self.model_name}/"
        )
        os.makedirs(target_dir, exist_ok=True)
        self.adb.pull(
            f"/data/local/tmp/{self.model_name}/output", f"{target_dir}/output"
        )

        return f"{target_dir}/output"

    def _flatten_dlc_outputs(self, dlc_matcher: Dict[str, str]) -> None:
        for _, result_path in dlc_matcher.items():
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

                    source_path = os.path.join(root, file)
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

    def _process_diagview_csv(self, csv_path: str):
        df = pd.read_csv(csv_path, index_col=False)

        filtered_df = df[df["Layer Id"].notna()]
        layer_stats = (
            filtered_df.groupby("Layer Id")
            .agg(
                {
                    "Time": ["mean"],
                    "Layer Name": "first",
                    "Unit of Measurement": "first",
                }
            )
            .reset_index()
        )

        layer_stats.columns = ["layer_id", "time_mean", "layer_name", "unit"]
        layer_stats["time_mean"] = layer_stats["time_mean"].round(0)
        layer_stats["time_mean"] = layer_stats["time_mean"].astype(int)
        layer_stats["layer_id"] = layer_stats["layer_id"].astype(int)
        layer_stats["layer_name"] = layer_stats["layer_name"].apply(
            lambda x: x.split(":")[0]
        )
        layer_stats["layer_name"] = self._replace_bad_layer_names(
            list(layer_stats["layer_name"])
        )
        total_time = layer_stats["time_mean"].sum()
        layer_stats["Percentage_of_Total_Time"] = (
            layer_stats["time_mean"] / total_time
        ).round(4)

        layer_stats.to_csv(
            f"{str(constants.OUTPUTS_DIR)}/analysis/{self.model_name}/layer_cycles.csv",
            index=False,
        )

    # cleanup
    def _cleanup_dlc_outputs(self) -> None:
        self.adb.shell(f"rm -rf /data/local/tmp/{self.model_name}")

        output_dir = Path(
            f"{str(constants.OUTPUTS_DIR)}/analysis/{self.model_name}/output"
        )
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
