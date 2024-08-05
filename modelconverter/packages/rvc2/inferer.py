from pathlib import Path
from typing import Dict

import numpy as np
from openvino import Core

from modelconverter.utils import read_image

from ..base_inferer import Inferer


class RVC2Inferer(Inferer):
    def setup(self):
        self.xml_path = self.model_path
        self.bin_path = self.model_path.with_suffix(".bin")
        ie = Core()
        model = ie.compile_model(model=self.xml_path, device_name="CPU")
        self.infer_request = model.create_infer_request()

    def infer(self, inputs: Dict[str, Path]) -> Dict[str, np.ndarray]:
        arr_inputs = {
            name: read_image(
                path,
                shape=self.in_shapes[name],
                encoding=self.encoding[name],
                resize_method=self.resize_method[name],
                data_type=self.in_dtypes[name],
            )[np.newaxis, ...]
            for name, path in inputs.items()
        }
        result = self.infer_request.infer(inputs=arr_inputs)
        return {
            next(iter(name)): result[next(iter(name))]
            for name in result.names()
        }
