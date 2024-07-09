from pathlib import Path
from typing import Dict

import numpy as np
from openvino.inference_engine.ie_api import IECore

from modelconverter.utils import read_image

from ..base_inferer import Inferer


class RVC2Inferer(Inferer):
    def setup(self):
        self.xml_path = self.model_path
        self.bin_path = self.model_path.with_suffix(".bin")
        ie = IECore()
        net = ie.read_network(model=self.xml_path, weights=self.bin_path)
        self.exec_net = ie.load_network(network=net, device_name="CPU")

    def infer(self, inputs: Dict[str, Path]) -> Dict[str, np.ndarray]:
        arr_inputs = {
            name: read_image(
                path,
                shape=self.in_shapes[name],
                encoding=self.encoding[name],
                resize_method=self.resize_method[name],
                data_type=self.in_dtypes[name],
            )
            for name, path in inputs.items()
        }
        return self.exec_net.infer(inputs=arr_inputs)
