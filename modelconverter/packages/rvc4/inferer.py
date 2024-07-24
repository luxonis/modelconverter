import shutil
from pathlib import Path
from typing import Dict

import numpy as np

from modelconverter.utils import read_image, subprocess_run

from ..base_inferer import Inferer


class RVC4Inferer(Inferer):
    def setup(self):
        self.raw_images_path = Path("raw_images")
        self.header = f"%{' '.join(name for name in self.out_shapes)}"

    def infer(self, inputs: Dict[str, Path]) -> Dict[str, np.ndarray]:
        outputs_path = Path("output")
        shutil.rmtree(self.raw_images_path, ignore_errors=True)
        self.raw_images_path.mkdir(parents=True)
        shutil.rmtree(outputs_path, ignore_errors=True)

        with open("input_list.txt", "w") as f:
            f.write(self.header + "\n")
            for input_name, path in inputs.items():
                raw_path = Path(f"raw_images/{input_name}.raw")
                arr = read_image(
                    path,
                    shape=self.in_shapes[input_name],
                    encoding=self.encoding[input_name],
                    resize_method=self.resize_method[input_name],
                    data_type=self.in_dtypes[input_name],
                    transpose=False,
                )
                arr.tofile(raw_path)
                f.write(f"{input_name}:={raw_path} ")
            f.write("\n")

        subprocess_run(
            [
                "snpe-net-run",
                "--container",
                str(self.model_path),
                "--input_list",
                "input_list.txt",
                "--output_dir",
                "output",
            ],
            silent=True,
        )
        out_paths = outputs_path.rglob("*.raw")
        outputs = {}
        for p in out_paths:
            arr = np.fromfile(
                p, dtype=self.out_dtypes[p.stem].as_numpy_dtype()
            )
            out_shape = self.out_shapes[p.stem]

            # TODO: detect layout
            if len(out_shape) == 4:
                N, C, H, W = out_shape
                outputs[p.stem] = arr.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            else:
                outputs[p.stem] = arr.reshape(out_shape)
        return outputs
