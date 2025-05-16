import contextlib
from io import StringIO
from pathlib import Path
from typing import cast

import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

from modelconverter.packages.base_inferer import Inferer
from modelconverter.packages.hailo.exporter import HailoExporter
from modelconverter.utils import read_image


class HailoInferer(Inferer):
    def setup(self) -> None:
        self.runner = ClientRunner(
            hw_arch=self.config.hailo.hw_arch
            if self.config is not None
            else "hailo8",
            har=str(self.model_path),
        )
        hn_dict = cast(dict, self.runner.get_hn_dict())
        output_hn_names = hn_dict["net_params"]["output_layers_order"]
        orig_meta = self.runner._original_model_meta
        if orig_meta is None:
            raise RuntimeError("Could not get original model metadata.")

        self.output_names = list(orig_meta["inverse_postprocess_io_map"])
        if len(self.output_names) > 1:
            raise NotImplementedError(
                "Multiple outputs are not supported at the moment."
            )
        for hn_name, params in hn_dict["layers"].items():
            if hn_name in output_hn_names:
                self.output_names.extend(params["original_names"])

    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]:
        stdout = stderr = StringIO()
        arr_inputs = {
            HailoExporter._get_hn_layer_info(self.runner, name)[0]: read_image(
                path,
                shape=self.in_shapes[name],
                encoding=self.encoding[name],
                resize_method=self.resize_method[name],
                data_type=self.in_dtypes[name],
            ).transpose(1, 2, 0)[np.newaxis, ...]
            for name, path in inputs.items()
        }

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.runner.infer_context(
                InferenceContext.SDK_QUANTIZED
            ) as context,
        ):
            outputs = self.runner.infer(
                context=context, dataset=arr_inputs, batch_size=1
            )
            return {
                self.output_names[idx]: output
                for idx, output in enumerate(outputs)
            }
