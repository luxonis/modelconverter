"""Inference with a Hailo model through the Hailo SDK.

Holds the `Inferer` implementation the ``infer`` command uses for the
Hailo platform: the converted HAR model is loaded with
``hailo_sdk_client`` and run in the SDK's quantized inference context.
It only works inside the Hailo Docker image, where that SDK is
installed.
"""

import contextlib
from io import StringIO
from pathlib import Path

import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

from modelconverter.platforms.base_inferer import Inferer
from modelconverter.platforms.hailo.exporter import HailoExporter
from modelconverter.utils import read_image


class HailoInferer(Inferer):
    """Inferer for Hailo HAR models based on the Hailo SDK client."""

    def setup(self) -> None:
        """Load the HAR model and collect the output names.

        The names are taken from the inverse postprocess map recorded
        in the model metadata, which only a HAR translated from ONNX
        carries, and from the original names of the output layers.

        Raises:
            RuntimeError: If the model carries no original metadata.
            NotImplementedError: If the inverse postprocess map holds
                more than one entry.

        """
        self._runner = ClientRunner(
            hw_arch=self.config.hailo.hw_arch
            if self.config is not None
            else "hailo8",
            har=str(self.model_path),
        )
        hn_dict = self._runner.get_hn_dict()
        output_hn_names = hn_dict["net_params"]["output_layers_order"]
        orig_meta = self._runner._original_model_meta
        if orig_meta is None:  # pragma: no cover
            raise RuntimeError("Could not get original model metadata.")

        # A HAR translated from ONNX carries this postprocess map; one
        # translated from TFLite does not, so fall back to the output layers.
        self._output_names = list(
            orig_meta.get("inverse_postprocess_io_map", [])
        )
        if len(self._output_names) > 1:
            raise NotImplementedError(
                "Multiple outputs are not supported at the moment."
            )
        for hn_name, params in hn_dict["layers"].items():
            if hn_name in output_hn_names:
                self._output_names.extend(params["original_names"])

    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]:
        """Run the model on a single set of input files.

        Every file is read according to the input it belongs to: an
        image is converted, while ``.npy`` and ``.raw`` data is loaded
        as is and must already be laid out as ``CHW``. The result is
        handed to the runner as a channels-last batch of one, keyed by
        the name of the matching HN layer. Whatever the SDK prints to
        stdout and stderr while inferring is discarded.

        Args:
            inputs: Path to the image, ``.npy`` or ``.raw`` file for
                every model input, keyed by input name.

        Returns:
            The model outputs, keyed by output name.

        """
        stdout = stderr = StringIO()
        arr_inputs = {
            HailoExporter._get_hn_layer_info(self._runner, name)[
                0
            ]: read_image(
                path,
                shape=self.in_shapes[name],
                encoding=self.encoding[name],
                resize_method=self.resize_method[name],
                data_type=self.in_dtypes[name],
                layout=self.layout.get(name),
            ).transpose(1, 2, 0)[np.newaxis, ...]
            for name, path in inputs.items()
        }

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self._runner.infer_context(
                InferenceContext.SDK_QUANTIZED
            ) as context,
        ):
            outputs = self._runner.infer(
                context=context, dataset=arr_inputs, batch_size=1
            )
            return {
                self._output_names[idx]: output
                for idx, output in enumerate(outputs)
            }
