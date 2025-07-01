from enum import Enum
from itertools import permutations

import numpy as np
from loguru import logger
from scipy.spatial.distance import cosine


class Metric(Enum):
    MAE = "mae"
    MSE = "mse"
    PSNR = "psnr"
    COS = "cos"
    SOFT_COS = "soft_cos"
    MAPE = "mape"
    MAX_PE = "max_abs_pe"  # maximum percentage error
    MIN_PE = "min_abs_pe"  # minimum percentage error

    @property
    def sign(self) -> str:
        if self in {self.PSNR, self.COS}:
            return ">="
        return "<="

    def compute(self, dlc: np.ndarray, onnx: np.ndarray) -> float:
        dlc, onnx = match_shapes_with_transpose(dlc, onnx)
        if self is self.MAE:
            return float(np.mean(np.abs(dlc - onnx)))
        if self is self.MSE:
            return float(np.mean((dlc - onnx) ** 2))
        if self is self.PSNR:
            mse = np.mean((dlc - onnx) ** 2)
            if mse == 0:
                return 1000  # Perfect match
            max_pixel = np.max([dlc.max(), onnx.max()])
            return 20 * np.log10(max_pixel) - 10 * np.log10(mse)
        if self is self.COS:
            return float(1 - cosine(dlc.flatten(), onnx.flatten()))
        if self is self.SOFT_COS:
            return float(
                1 - cosine(dlc.flatten() + 1e-5, onnx.flatten() + 1e-5)
            )
        raise ValueError(
            f"Unsupported metric: {self}. Supported metrics are: {', '.join(m.value for m in Metric)}"
        )

    def verify(self, old_score: float, new_score: float) -> None:
        if self in {self.COS, self.PSNR}:
            if old_score > new_score:
                raise RuntimeError(
                    f"Degradation test failed: old model has higher {self.value}  ({old_score}) than new model ({new_score})"
                )
        elif old_score < new_score:
            raise RuntimeError(
                f"Degradation test failed: old model has lower {self.value}  ({old_score}) than new model ({new_score})"
            )


def match_shapes_with_transpose(
    dlc: np.ndarray, onnx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if dlc.shape == onnx.shape:
        return dlc, onnx

    if dlc.ndim != onnx.ndim:
        raise ValueError(
            "Arrays have different number of dimensions, cannot transpose to match."
        )

    # brute-force, but it doesn't matter for small dimensions
    for perm in permutations(range(dlc.ndim)):
        dlc_transposed = dlc.transpose(perm)
        if onnx.shape == dlc_transposed.shape:
            logger.warning(
                f"Transposing dlc output from shape {dlc.shape} to {dlc_transposed.shape} to match the onnx output shape."
            )
            return dlc_transposed, onnx

    raise ValueError(
        f"The shape of the dlc output {dlc.shape} is not compatible with the onnx output shape {onnx.shape}. "
    )
