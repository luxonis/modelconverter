from enum import Enum

import numpy as np
from scipy.spatial.distance import cosine


class Metric(Enum):
    MAE = "mae"
    MSE = "mse"
    PSNR = "psnr"
    COS = "cos"

    @property
    def sign(self) -> str:
        if self in {self.PSNR, self.COS}:
            return ">="
        return "<="

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        if self is self.MAE:
            return float(np.mean(np.abs(a - b)))
        if self is self.MSE:
            return float(np.mean((a - b) ** 2))
        if self is self.PSNR:
            mse = np.mean((a - b) ** 2)
            if mse == 0:
                return 1000  # Perfect match
            max_pixel = np.max([a.max(), b.max()])
            return 20 * np.log10(max_pixel) - 10 * np.log10(mse)
        if self is self.COS:
            return float(1 - cosine(a.flatten(), b.flatten()))
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
