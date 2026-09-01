"""Reference inference on the *original* ONNX model, via onnxruntime.

modelconverter *bakes* the preprocessing (channel reversal + mean/scale) into the
converted model, so a converted model is fed a raw image in its ``to`` encoding
and normalizes internally. The original ONNX has no such layer, so to get a
comparable output we apply the same transform here first: the ``from`` encoding
image, normalized as ``(x - mean) / scale`` -- what
``onnx_tools.add_preprocessing`` inserts.

Host-side and dependency-light (onnxruntime + numpy) on purpose, so it runs
anywhere the unit tests do, unlike the vendor inferers.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from luxonis_ml.typing import PathType

from modelconverter.utils.config import (
    ImageCalibrationConfig,
    InputConfig,
    SingleStageConfig,
)
from modelconverter.utils.image import read_image
from modelconverter.utils.types import DataType, Encoding, ResizeMethod


class ONNXReferenceInferer:
    """Runs the original ONNX model with the config's preprocessing applied.

    Build with ``from_stage``, then call ``infer`` with an image path to get
    ``{output_name: array}`` -- the outputs the converted model should reproduce.
    """

    def __init__(self, model_path: Path, stage: SingleStageConfig) -> None:
        """Open an ONNX session for the stage's model.

        Args:
            model_path: Path to the ONNX model to run.
            stage: Stage config whose preprocessing is applied to the
                inputs before inference.

        """
        self.stage = stage
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

    @classmethod
    def from_stage(cls, stage: SingleStageConfig) -> "ONNXReferenceInferer":
        """Build an inferer for the stage's own input model.

        Args:
            stage: Stage config to take the model and preprocessing from.

        Returns:
            An inferer for that stage.

        """
        # `get_configs` downloads `input_model` and rewrites it to a local path.
        return cls(Path(stage.input_model), stage)

    @staticmethod
    def _resize_method(inp: InputConfig) -> ResizeMethod:
        if isinstance(inp.calibration, ImageCalibrationConfig):
            return inp.calibration.resize_method
        return ResizeMethod.RESIZE

    def _preprocess(
        self, inp: InputConfig, image_path: PathType
    ) -> np.ndarray:
        # Feed the original model its native (`from`) encoding, matching what
        # the baked reverse-channel step produces inside the converted model.
        # read_image yields CHW float in [0, 255], unnormalized.
        assert inp.shape is not None, f"input {inp.name!r} has no shape"
        channels_last = bool(inp.layout) and inp.layout[-1] == "C"
        arr = read_image(
            image_path,
            inp.shape,
            inp.encoding.from_,
            self._resize_method(inp),
            data_type=DataType.FLOAT32,
            transpose=not channels_last,
            layout=inp.layout,
        ).astype(np.float32)

        # Planar data needs mean/scale broadcast over the leading channel axis;
        # channels-last and raw data broadcast over the trailing one.
        planar = not channels_last and inp.encoding.from_ is not Encoding.NONE
        if inp.mean_values is not None and any(inp.mean_values):
            mean = np.asarray(inp.mean_values, dtype=np.float32)
            arr = arr - (mean.reshape(-1, 1, 1) if planar else mean)
        if inp.scale_values is not None and any(
            v != 1 for v in inp.scale_values
        ):
            scale = np.asarray(inp.scale_values, dtype=np.float32)
            arr = arr / (scale.reshape(-1, 1, 1) if planar else scale)

        return arr[np.newaxis, ...]

    def infer(self, image_path: PathType) -> dict[str, np.ndarray]:
        """Run the model on one image, preprocessed per the config.

        Args:
            image_path: Path to the image to run on. It is fed to every
                input, each preprocessed with that input's own settings.

        Returns:
            Mapping of output name to output array.

        """
        # Config inputs pair with the session inputs positionally.
        feed = {
            name: self._preprocess(inp, image_path)
            for name, inp in zip(
                self._input_names, self.stage.inputs, strict=True
            )
        }
        outputs = self.session.run(self._output_names, feed)
        return dict(
            zip(self._output_names, map(np.asarray, outputs), strict=True)
        )
