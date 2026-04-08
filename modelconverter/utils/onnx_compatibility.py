from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
from onnx.external_data_helper import convert_model_to_external_data


def ensure_onnx_helper_compatibility() -> None:
    helper = onnx.helper

    def _convert_scalar(
        value: float, dtype: np.dtype, container: np.dtype
    ) -> int:
        arr = np.asarray(value, dtype=dtype)
        return arr.view(container).item()

    if not hasattr(helper, "float32_to_bfloat16"):
        helper.float32_to_bfloat16 = lambda value: _convert_scalar(  # type: ignore[attr-defined]
            value, ml_dtypes.bfloat16, np.uint16
        )

    if not hasattr(helper, "float32_to_float8e4m3"):
        dtype_map = {
            (False, False): ml_dtypes.float8_e4m3,
            (True, False): ml_dtypes.float8_e4m3fn,
            (True, True): ml_dtypes.float8_e4m3fnuz,
            (False, True): ml_dtypes.float8_e4m3b11fnuz,
        }

        def float32_to_float8e4m3(
            value: float, *, fn: bool = True, uz: bool = False
        ) -> int:
            return _convert_scalar(value, dtype_map[(fn, uz)], np.uint8)

        helper.float32_to_float8e4m3 = float32_to_float8e4m3  # type: ignore[attr-defined]


def save_onnx_model(
    model: onnx.ModelProto,
    output_path: str | Path,
    *,
    save_as_external_data: bool = False,
    location: str | None = None,
) -> None:
    output_path = Path(output_path)

    if save_as_external_data:
        external_data_path = output_path.with_name(
            location or f"{output_path.name}_data"
        )
        if external_data_path.exists():
            external_data_path.unlink()
        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
            size_threshold=0,
            convert_attribute=False,
        )
        onnx.save(
            model,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
            size_threshold=0,
            convert_attribute=False,
        )
        return

    onnx.save(model, str(output_path))
