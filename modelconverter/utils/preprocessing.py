"""Calibration-side representation of externalized input preprocessing.

Moving the preprocessing to an NN Archive clears it from the conversion
config, but calibration still needs it: the quantizer then sees the input of
the source model, not of a model that carries the preprocessing nodes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from modelconverter.utils.exceptions import ModelconverterException
from modelconverter.utils.types import DataType, Encoding

if TYPE_CHECKING:
    from modelconverter.utils.config import ImageCalibrationConfig


def normalization_required(
    mean_values: Sequence[float] | None,
    scale_values: Sequence[float] | None,
) -> bool:
    """Whether mean subtraction or scale division is non-identity."""
    return (
        mean_values is not None and any(value != 0 for value in mean_values)
    ) or (
        scale_values is not None and any(value != 1 for value in scale_values)
    )


def input_preprocessing_required(
    *,
    encoding_from: Encoding,
    encoding_to: Encoding,
    mean_values: Sequence[float] | None,
    scale_values: Sequence[float] | None,
    reverse_only: bool = False,
) -> bool:
    """Whether an input's color conversion or normalization is non-identity."""
    if encoding_from != encoding_to:
        return True
    return not reverse_only and normalization_required(
        mean_values, scale_values
    )


@dataclass(frozen=True)
class CalibrationPreprocessing:
    """Immutable preprocessing needed only by calibration preparation."""

    encoding_from: Encoding
    encoding_to: Encoding
    mean_values: tuple[float, ...] | None
    scale_values: tuple[float, ...] | None
    data_type: DataType
    is_image: bool

    @property
    def encoding_mismatch(self) -> bool:
        """Whether the source model and its preprocessed input differ."""
        return self.encoding_from != self.encoding_to

    @property
    def normalization_required(self) -> bool:
        """Whether mean subtraction or scale division is non-identity."""
        return normalization_required(self.mean_values, self.scale_values)

    def requires_input_preprocessing(
        self, *, reverse_only: bool = False
    ) -> bool:
        """Whether this snapshot contains non-identity preprocessing."""
        return input_preprocessing_required(
            encoding_from=self.encoding_from,
            encoding_to=self.encoding_to,
            mean_values=self.mean_values,
            scale_values=self.scale_values,
            reverse_only=reverse_only,
        )


def apply_calibration_preprocessing(
    array: np.ndarray,
    preprocessing: CalibrationPreprocessing,
    *,
    layout: str,
) -> np.ndarray:
    """Apply externalized color conversion and normalization to an array.

    ``layout`` describes the array as it exists at this point. It may omit a
    singleton batch axis even when the configured model layout has one, which
    is how a decoded image normally arrives.
    """
    if len(layout) != array.ndim:
        raise ModelconverterException(
            f"Calibration array with shape {list(array.shape)} cannot use "
            f"layout '{layout}'."
        )

    result = np.asarray(array)
    channel_axis = layout.find("C")

    if preprocessing.encoding_mismatch:
        if channel_axis < 0 or result.shape[channel_axis] != 3:
            raise ModelconverterException(
                "Calibration channel reversal requires an array with exactly "
                f"three channels, got shape {list(result.shape)} and layout "
                f"'{layout}'."
            )
        result = np.flip(result, axis=channel_axis)

    if preprocessing.normalization_required:
        target_dtype = preprocessing.data_type.as_numpy_dtype()
        if not np.issubdtype(target_dtype, np.floating):
            raise ModelconverterException(
                "Externalized mean/scale calibration requires a floating-point "
                f"model input, but the input data type is "
                f"'{preprocessing.data_type.value}'."
            )
        if channel_axis < 0:
            raise ModelconverterException(
                "Externalized mean/scale calibration requires a layout with a "
                f"channel axis, got '{layout}'."
            )

        # A float16 model input still normalizes in float32; only a float64
        # one needs the wider accumulator.
        calculation_dtype = (
            np.float64 if target_dtype == np.float64 else np.float32
        )
        channels = result.shape[channel_axis]
        mean = _per_channel(
            preprocessing.mean_values, channels, "mean", calculation_dtype
        )
        scale = _per_channel(
            preprocessing.scale_values, channels, "scale", calculation_dtype
        )
        broadcast_shape = [1] * result.ndim
        broadcast_shape[channel_axis] = channels

        result = result.astype(calculation_dtype, copy=False)
        if mean is not None:
            result = result - mean.reshape(broadcast_shape)
        if scale is not None:
            result = result / scale.reshape(broadcast_shape)

    return result.astype(preprocessing.data_type.as_numpy_dtype(), copy=False)


def reorder_layout(
    array: np.ndarray, source_layout: str, target_layout: str
) -> np.ndarray:
    """Reorder axes, adding or removing only a singleton batch dimension."""
    result = array
    current = source_layout

    if "N" in target_layout and "N" not in current:
        axis = target_layout.index("N")
        result = np.expand_dims(result, axis=axis)
        current = current[:axis] + "N" + current[axis:]
    elif "N" in current and "N" not in target_layout:
        axis = current.index("N")
        if result.shape[axis] != 1:
            raise ModelconverterException(
                "Only a singleton calibration batch can be removed, got "
                f"shape {list(result.shape)}."
            )
        result = np.squeeze(result, axis=axis)
        current = current[:axis] + current[axis + 1 :]

    if sorted(current) != sorted(target_layout):
        raise ModelconverterException(
            f"Cannot reorder calibration layout '{source_layout}' to "
            f"'{target_layout}'."
        )
    return result.transpose([current.index(axis) for axis in target_layout])


def channels_last_image_layout(layout: str) -> str:
    """Return the NHWC/HWC counterpart of an image layout.

    A layout that does not describe an image is returned unchanged.
    """
    if sorted(layout) == ["C", "H", "N", "W"]:
        return "NHWC"
    if sorted(layout) == ["C", "H", "W"]:
        return "HWC"
    return layout


def channels_last_4d_layout(layout: str) -> str:
    """Move a unique channel axis last in a four-dimensional layout."""
    if len(layout) == 4 and layout.count("C") == 1:
        return layout.replace("C", "") + "C"
    return layout


def is_user_calibration_tensor(
    path: Path, calib: "ImageCalibrationConfig"
) -> bool:
    """Whether a tensor file uses the backend-ready calibration contract."""
    return (
        path.suffix.lower() in {".npy", ".raw"}
        and not calib.generated_from_random
    )


def read_user_calibration_tensor(
    path: Path,
    *,
    raw_shape: list[int] | tuple[int, ...] | None = None,
    data_type: DataType,
    input_name: str,
) -> np.ndarray:
    """Load an opaque user tensor in a backend-required sample shape.

    NumPy files carry their own shape and are returned unchanged. Raw buffers
    require ``raw_shape`` and are reshaped without reordering after their
    element count is validated.
    """
    if path.suffix.lower() == ".npy":
        return np.load(path)

    if raw_shape is None:
        raise ValueError("A shape is required to load a raw calibration file.")
    array = np.fromfile(path, dtype=data_type.as_numpy_dtype())
    expected_size = int(np.prod(raw_shape))
    if array.size != expected_size:
        raise ModelconverterException(
            f"Calibration data for input '{input_name}' has {array.size} "
            f"elements, expected {expected_size} for shape {list(raw_shape)}."
        )
    return array.reshape(raw_shape)


def _per_channel(
    values: tuple[float, ...] | None,
    channels: int,
    name: str,
    dtype: type[np.float32] | type[np.float64],
) -> np.ndarray | None:
    """Expand preprocessing values to one value per channel."""
    if values is None:
        return None
    if len(values) == 1:
        values = values * channels
    if len(values) != channels:
        raise ModelconverterException(
            f"Calibration {name} has {len(values)} values for an array with "
            f"{channels} channels."
        )
    return np.asarray(values, dtype=dtype)
