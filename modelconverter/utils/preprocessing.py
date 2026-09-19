"""Calibration-side representation of externalized input preprocessing.

Moving the preprocessing to an NN Archive clears it from the conversion
config, but calibration still needs it: the quantizer then sees the input of
the source model, not of a model that carries the preprocessing nodes.
"""

from dataclasses import dataclass

import numpy as np

from modelconverter.utils.exceptions import ModelconverterException
from modelconverter.utils.types import DataType, Encoding


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
        return (
            self.mean_values is not None
            and any(value != 0 for value in self.mean_values)
        ) or (
            self.scale_values is not None
            and any(value != 1 for value in self.scale_values)
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


def array_layout(
    array: np.ndarray,
    *,
    shape: list[int],
    layout: str | None,
) -> str:
    """Resolve the configured layout for an array with optional batch omitted."""
    if layout is None:
        raise ModelconverterException(
            "Calibration preprocessing requires the input layout to be known."
        )
    if array.ndim == len(layout):
        return layout

    if (
        array.ndim + 1 == len(layout)
        and "N" in layout
        and shape[layout.index("N")] == 1
    ):
        return layout.replace("N", "", 1)

    raise ModelconverterException(
        f"Calibration array shape {list(array.shape)} is incompatible with "
        f"input layout '{layout}'."
    )


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
