"""Reference inference on the *original* ONNX model (onnxruntime).

Used by the conversion precision tests to obtain a ground-truth output to
compare a converted model against. It is deliberately host-side and
dependency-light (onnxruntime + numpy) so it runs anywhere the unit tests
do -- unlike the vendor inferers, which need their Docker runtimes.

The subtlety it exists to handle: ``modelconverter`` *bakes* the
preprocessing (channel reversal + mean/scale normalization) into the
converted model, so a converted model is fed a raw image (in its ``to``
encoding) and normalizes internally. The original ONNX has no such layer,
so to get a comparable output we must apply that same preprocessing here
before running it -- feeding the original model the ``from`` encoding image
normalized as ``(x - mean) / scale`` (the exact transform
``onnx_tools.add_preprocessing`` inserts, see that function).
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from modelconverter.utils.config import (
    ImageCalibrationConfig,
    InputConfig,
    SingleStageConfig,
)
from modelconverter.utils.image import read_image
from modelconverter.utils.types import DataType, Encoding, ResizeMethod


class ONNXReferenceInferer:
    """Runs the original ONNX model with the config's preprocessing applied.

    Build with `from_stage`, then call `infer` with an image
    path to get ``{output_name: np.ndarray}`` -- the reference outputs the
    converted model should reproduce.
    """

    def __init__(self, model_path: Path, stage: SingleStageConfig) -> None:
        self.stage = stage
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

    @classmethod
    def from_stage(cls, stage: SingleStageConfig) -> "ONNXReferenceInferer":
        # `get_configs` downloads `input_model` and rewrites it to a local
        # path (see config.SingleStageConfig._download_input_model).
        return cls(Path(stage.input_model), stage)

    @staticmethod
    def _resize_method(inp: InputConfig) -> ResizeMethod:
        if isinstance(inp.calibration, ImageCalibrationConfig):
            return inp.calibration.resize_method
        return ResizeMethod.RESIZE  # pragma: no cover

    def _preprocess(
        self, inp: InputConfig, image_path: str | Path
    ) -> np.ndarray:
        # Feed the *original* model its native (`from`) encoding, matching
        # what the baked reverse-channel step produces inside the converted
        # model. read_image yields CHW float in [0, 255] (no normalization).
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

        channel_axis = (
            -1
            if channels_last
            else 0
            if inp.encoding.from_ != Encoding.NONE
            else None
        )
        if inp.mean_values is not None and any(inp.mean_values):
            mean = np.asarray(inp.mean_values, dtype=np.float32)
            arr = (
                arr - mean.reshape(-1, 1, 1)
                if channel_axis == 0
                else arr - mean
            )
        if inp.scale_values is not None and any(
            v != 1 for v in inp.scale_values
        ):
            scale = np.asarray(inp.scale_values, dtype=np.float32)
            arr = (
                arr / scale.reshape(-1, 1, 1)
                if channel_axis == 0
                else arr / scale
            )

        return arr[np.newaxis, ...]  # add batch dim -> NCHW

    def infer(self, image_path: str | Path) -> dict[str, np.ndarray]:
        # Pair config inputs with the session inputs positionally (same order).
        feed = {
            name: self._preprocess(inp, image_path)
            for name, inp in zip(
                self._input_names, self.stage.inputs, strict=True
            )
        }
        outputs = self.session.run(self._output_names, feed)
        return {
            name: np.asarray(output)
            for name, output in zip(self._output_names, outputs, strict=True)
        }
