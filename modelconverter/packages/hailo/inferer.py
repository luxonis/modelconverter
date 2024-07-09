import contextlib
from io import StringIO
from logging import getLogger
from pathlib import Path
from typing import Dict

import numpy as np

from modelconverter.packages.hailo.exporter import (
    HailoExporter,
    _replace_pydantic,
)
from modelconverter.utils import read_image

from ..base_inferer import Inferer

logger = getLogger(__name__)

with _replace_pydantic():
    from hailo_sdk_client import ClientRunner, InferenceContext


class HailoInferer(Inferer):
    def setup(self):
        self.runner = ClientRunner(
            hw_arch="hailo8",
            har=str(self.model_path),
        )
        hn_dict = self.runner.get_hn_dict()
        output_hn_names = hn_dict["net_params"]["output_layers_order"]
        orig_meta = self.runner._original_model_meta
        if orig_meta is None:
            raise RuntimeError("Could not get original model metadata.")

        self.output_names = [
            k for k in orig_meta["inverse_postprocess_io_map"]
        ]
        if len(self.output_names) > 1:
            raise NotImplementedError(
                "Multiple outputs are not supported at the moment."
            )
        for hn_name, params in hn_dict["layers"].items():  # type: ignore
            if hn_name in output_hn_names:
                self.output_names.extend(params["original_names"])

    def infer(self, inputs: Dict[str, Path]) -> Dict[str, np.ndarray]:
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

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            stderr
        ):
            with self.runner.infer_context(
                InferenceContext.SDK_QUANTIZED
            ) as context:
                outputs = self.runner.infer(
                    context=context, dataset=arr_inputs, batch_size=1
                )
                return {
                    self.output_names[idx]: output
                    for idx, output in enumerate(outputs)
                }
