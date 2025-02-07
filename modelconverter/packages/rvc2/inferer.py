import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

from modelconverter.utils import read_image, subprocess_run

from ..base_inferer import Inferer


class RVC2Inferer(Inferer):
    def setup(self):
        self.xml_path = self.model_path
        self.bin_path = self.model_path.with_suffix(".bin")

    def infer(self, inputs: Dict[str, Path]) -> Dict[str, np.ndarray]:
        args = ["ov_infer", "--xml-path", self.xml_path]

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

        with tempfile.TemporaryDirectory() as temp_dir:
            for name, arr in arr_inputs.items():
                path = (Path(temp_dir) / name).with_suffix(".npy")
                np.save(path, arr)
                args.extend(["--input", name, path])

            args.extend(["--out-path", temp_dir])

            subprocess_run(args)

            return {
                path.stem: np.load(path) for path in Path(temp_dir).iterdir()
            }
