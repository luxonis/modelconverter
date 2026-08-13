from pathlib import Path

import numpy as np

from modelconverter.platforms.base_inferer import Inferer
from modelconverter.utils import read_image


class RVC2Inferer(Inferer):
    def setup(self) -> None:
        from openvino.inference_engine.ie_api import IECore

        self.xml_path = self.model_path
        self.bin_path = self.model_path.with_suffix(".bin")
        ie = IECore()
        net = ie.read_network(model=self.xml_path, weights=self.bin_path)
        # Both the shape and the layout come from the IR. The config describes
        # the original model, whose layout the conversion need not keep, and
        # `read_image` reads the one through the other -- taking a shape here
        # and a layout there would feed the network a transposed image.
        for name, input_info in net.input_info.items():
            shape = list(input_info.input_data.shape)
            self.in_shapes[name] = shape
            if len(shape) == 4:
                self.layout[name] = "NCHW" if shape[1] in {1, 3, 4} else "NHWC"
        self.exec_net = ie.load_network(network=net, device_name="CPU")

    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]:
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
        return self.exec_net.infer(inputs=arr_inputs)
