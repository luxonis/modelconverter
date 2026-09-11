"""Inference with an RVC4 model through SNPE.

Holds the `Inferer` implementation the ``infer`` command uses for the
RVC4 platform: the inputs are written out as raw files and pushed
through the converted DLC model with ``snpe-net-run``. It only works
inside the RVC4 Docker image, where the SNPE SDK is installed.
"""

import shutil
from pathlib import Path

import numpy as np

from modelconverter.platforms.base_inferer import Inferer
from modelconverter.utils import read_image, subprocess_run
from modelconverter.utils.types import DataType


class RVC4Inferer(Inferer):
    """Inferer for RVC4 DLC models based on ``snpe-net-run``."""

    _output_layouts: dict[str, str | None]

    def setup(self) -> None:
        """Set paths and cache metadata used by every inference.

        The header names the outputs SNPE is asked to write out. Output layouts
        are captured once alongside the shapes and data types populated by
        ``Inferer.from_config``.

        """
        self._raw_images_path = Path("raw_images")
        self._header = f"%{' '.join(name for name in self.out_shapes)}"
        self._output_layouts = (
            {out.name: out.layout for out in self.config.outputs}
            if self.config is not None
            else {}
        )

    def infer(self, inputs: dict[str, Path]) -> dict[str, np.ndarray]:
        """Run the model on a single set of input images.

        Every image is read as ``float32`` and dumped to a raw file
        referenced from the SNPE input list, ``snpe-net-run`` is then
        invoked on the DLC model, and the raw files it produces are
        read back and reshaped to the configured output shapes.
        SNPE's four-dimensional output data is channels-last. Its
        dimensions are recovered from the configured output layout and
        exposed as ``NCHW``.

        Args:
            inputs: Path to the image for every model input, keyed by
                input name.

        Returns:
            The model outputs, keyed by output name.

        """
        # Scratch directory for SNPE's raw outputs. Must NOT be literally
        # "output": inside the container that resolves to `/app/output`, which
        # is the bind-mounted results directory the inferer writes into. Using
        # it here would wipe previously-saved results on every image.
        outputs_path = Path("snpe_output")
        shutil.rmtree(self._raw_images_path, ignore_errors=True)
        self._raw_images_path.mkdir(parents=True)
        shutil.rmtree(outputs_path, ignore_errors=True)

        with open("input_list.txt", "w") as f:
            f.write(self._header + "\n")
            for input_name, path in inputs.items():
                raw_path = Path(f"raw_images/{input_name}.raw")
                arr = read_image(
                    path,
                    shape=self.in_shapes[input_name],
                    encoding=self.encoding[input_name],
                    resize_method=self.resize_method[input_name],
                    data_type=DataType.FLOAT32,
                    transpose=False,
                    layout=self.layout.get(input_name),
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
                str(outputs_path),
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
            outputs[p.stem] = _reshape_output(
                arr, out_shape, self._output_layouts.get(p.stem)
            )
        return outputs


def _reshape_output(
    arr: np.ndarray, shape: list[int], layout: str | None
) -> np.ndarray:
    """Reshape an SNPE output, exposing four-dimensional tensors as NCHW."""
    if len(shape) != 4:
        return arr.reshape(shape)

    if layout == "NHWC":
        n, h, w, c = shape
    else:
        n, c, h, w = shape
    return arr.reshape(n, h, w, c).transpose(0, 3, 1, 2)
