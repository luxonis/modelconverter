"""Inference with models converted for the RVC2 platform.

RVC2 conversions produce an OpenVINO IR (an ``.xml`` topology next to a
``.bin`` weights file). This module runs such an IR on the CPU through
the OpenVINO inference engine, which is available inside the RVC2
container, so that the converted model can be fed the same images as the
original one.
"""

from pathlib import Path

import numpy as np

from modelconverter.platforms.base_inferer import Inferer
from modelconverter.utils import read_image


class RVC2Inferer(Inferer):
    """Inferer for RVC2 models, using the OpenVINO inference engine."""

    def setup(self) -> None:
        """Load the IR model and prepare it for inference on the CPU.

        Reads the ``.xml`` topology together with the ``.bin`` weights
        sitting next to it, takes the input shapes from the IR and
        derives the layout of every 4-D input from where its channel
        dimension sits, then loads the network onto the CPU.
        """
        from openvino.inference_engine.ie_api import IECore

        self._xml_path = self.model_path
        self._bin_path = self.model_path.with_suffix(".bin")
        ie = IECore()
        net = ie.read_network(model=self._xml_path, weights=self._bin_path)
        # Both the shape and the layout come from the IR. The config describes
        # the original model, whose layout the conversion need not keep, and
        # `read_image` reads the one through the other -- taking a shape here
        # and a layout there would feed the network a transposed image.
        for name, input_info in net.input_info.items():
            shape = list(input_info.input_data.shape)
            self.in_shapes[name] = shape
            if len(shape) == 4:
                self.layout[name] = "NCHW" if shape[1] in {1, 3, 4} else "NHWC"
        self._exec_net = ie.load_network(network=net, device_name="CPU")

    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]:
        """Run the network on one image per input.

        Every image is read in the layout the network expects and gets a
        batch dimension prepended when it comes out one dimension short
        of the input shape.

        Args:
            inputs: Mapping from input name to the image file to feed
                to it.

        Returns:
            Mapping from output name to the array the network produced.

        """
        arr_inputs = {}
        for name, path in inputs.items():
            layout = self.layout.get(name)
            channels_last = bool(layout) and layout[-1] == "C"
            image = read_image(
                path,
                shape=self.in_shapes[name],
                encoding=self.encoding[name],
                resize_method=self.resize_method[name],
                data_type=self.in_dtypes[name],
                # Feed the input in the network's own layout: channels-last for
                # an NHWC (e.g. TFLite-derived) IR, channels-first otherwise.
                transpose=not channels_last,
                layout=layout,
            )
            if image.ndim == len(self.in_shapes[name]) - 1:
                image = image[np.newaxis, ...]
            arr_inputs[name] = image
        return self._exec_net.infer(inputs=arr_inputs)
