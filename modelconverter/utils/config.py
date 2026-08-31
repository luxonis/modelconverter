"""Configuration models describing a conversion.

A conversion is described by a YAML file (see ``configs/defaults.yaml``)
that is parsed into the pydantic models defined here. The top-level
`Config` holds one `SingleStageConfig` per stage, each describing the
input model, its inputs and outputs, the calibration data and the
platform-specific options (`HailoConfig`, `RVC2Config`, `RVC3Config` and
`RVC4Config`).

Values that the user did not provide are filled in from the metadata of
the input model itself, so that the exporter running inside the
per-backend Docker image receives a fully resolved description of the
model.
"""

import warnings
from itertools import chain
from pathlib import Path
from typing import Annotated, Literal

import onnx
from loguru import logger
from luxonis_ml.typing import (
    BaseModelExtraForbid,
    Params,
    ParamValue,
    PathType,
)
from luxonis_ml.utils import LuxonisConfig
from onnx import TypeProto
from pydantic import (
    Field,
    PositiveInt,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from modelconverter.utils.calibration_data import download_calibration_data
from modelconverter.utils.constants import MISC_DIR, MODELS_DIR
from modelconverter.utils.encodings import parse_encodings
from modelconverter.utils.filesystem_utils import resolve_path
from modelconverter.utils.layout import make_default_layout
from modelconverter.utils.metadata import Metadata, get_metadata
from modelconverter.utils.onnx_compatibility import (
    has_external_data,
    save_onnx_model,
)
from modelconverter.utils.types import (
    DataType,
    Encoding,
    InputFileType,
    Platform,
    PotDevice,
    QuantizationMode,
    ResizeMethod,
)

NAMED_VALUES = {
    "imagenet": {
        "mean": [123.675, 116.28, 103.53],
        "scale": [58.395, 57.12, 57.375],
    },
}


class LinkCalibrationConfig(BaseModelExtraForbid):
    """Calibration data produced by another stage of the conversion.

    Used for multi-stage models, where the calibration data of one stage
    is obtained by running inference with a previously converted stage.

    Attributes:
        stage: Name of the stage to take the calibration data from.
        output: Name of the output of ``stage`` to use as the
            calibration data.
        script: Python script post-processing the outputs of ``stage``.
            Either the source code itself or a path to a ``.py`` file,
            whose contents are read during validation. Either ``output``
            or ``script`` must be provided.

    """

    stage: str
    output: str | None = None
    script: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.output is None and self.script is None:
            raise ValueError(
                "Either `output` or `script` must be provided for calibration."
            )
        return self

    @field_validator("script", mode="after")
    @staticmethod
    def _download_calibration_script(script: str | None) -> str | None:
        if script is None:
            return None
        if script.endswith(".py"):
            script_path = resolve_path(script, MODELS_DIR)
            script = script_path.read_text()
        return script


class ImageCalibrationConfig(BaseModelExtraForbid):
    """Calibration data read from a directory of files.

    Attributes:
        path: Path to the calibration data. Can be a local directory, a
            remote URL, or an LDF dataset identifier in the
            ``<dataset_name>:<split>`` form; it is downloaded during
            validation and stored as a local path.
        max_images: Number of files to use from the calibration data.
            A negative value means all of them.
        resize_method: How to resize the images to the input shape.

    """

    path: Path
    max_images: int = -1
    resize_method: ResizeMethod = ResizeMethod.RESIZE

    @field_validator("path", mode="before")
    @staticmethod
    def _download_calibration_data(value: ParamValue) -> Path | None:
        if value is None:
            return None
        return download_calibration_data(str(value))


class RandomCalibrationConfig(BaseModelExtraForbid):
    """Calibration data generated from a normal distribution.

    Used when no calibration data is provided. The generated values are
    drawn from a normal distribution and clipped to the allowed range.

    Attributes:
        max_images: Number of samples to generate.
        min_value: Lower bound the generated values are clipped to.
        max_value: Upper bound the generated values are clipped to.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        data_type: Data type the generated samples are cast to.

    """

    max_images: int = 20
    min_value: float = 0.0
    max_value: float = 255.0
    mean: float = 127.5
    std: float = 35.0
    data_type: DataType = DataType.FLOAT32


class OutputConfig(BaseModelExtraForbid):
    """Description of a single output of the model.

    Attributes:
        name: Name of the output tensor or node.
        shape: Shape of the output. Inferred from the model if not
            provided.
        layout: Lettercode representation of the layout, e.g. ``"NCHW"``.
            Requires ``shape`` to be provided; a default layout is
            guessed from the shape if only ``shape`` is given.
        data_type: Data type of the output.

    """

    name: str
    shape: list[int] | None = None
    layout: str | None = None
    data_type: DataType = DataType.FLOAT32

    @model_validator(mode="before")
    @classmethod
    def _make_default_layout(cls, data: Params) -> Params:
        shape = data.get("shape")
        layout = data.get("layout")
        if shape is None and layout is not None:
            raise ValueError("`layout` cannot be provided without `shape`.")
        if shape is None:
            return data
        if layout is None:
            layout = make_default_layout(_as_shape(shape, "shape"))
        elif not isinstance(layout, str):
            raise TypeError("`layout` must be a string.")
        data["layout"] = layout.upper()
        return data

    @model_validator(mode="after")
    def validate_layout(self) -> Self:
        """Check that the layout and the shape have the same length.

        Returns:
            The validated model.

        Raises:
            ValueError: If the length of the layout does not match the
                length of the shape.

        """
        if self.shape is None:
            return self
        assert self.layout is not None
        if len(self.layout) != len(self.shape):
            raise ValueError(
                f"Length of `layout` ({len(self.layout)}) must match "
                f"length of `shape` ({len(self.shape)})."
            )
        return self


class EncodingConfig(BaseModelExtraForbid):
    """Color encoding conversion requested for an input.

    The ``from`` field, named ``from_`` in Python, is the encoding the
    original model expects. The ``to`` field is the encoding the
    converted model is fed with at runtime. If the two differ, a
    channel reversal is prepended to the input, so that the converted
    model takes ``to`` data while the original graph still receives
    ``from`` data.
    """

    from_: Encoding = Field(
        default=Encoding.RGB, alias="from", serialization_alias="from"
    )
    to: Encoding = Encoding.BGR


class InputConfig(OutputConfig):
    """Description of a single input of the model.

    Extends `OutputConfig` with the pre-processing and calibration
    settings of the input. ``mean_values`` and ``scale_values`` are the
    normalization `onnx_attach_normalization_to_inputs` bakes into the
    graph.

    Attributes:
        calibration: Source of the calibration data for this input.
        scale_values: Per-channel values the input is divided by, after
            ``mean_values`` has been subtracted. Can be given as a
            single number, a list, or the name of a preset such as
            ``"imagenet"``.
        mean_values: Per-channel values subtracted from the input, with
            the same options as ``scale_values``.
        frozen_value: List of constant values the input is frozen to.
            Used only by the OpenVINO-based conversions (RVC2 and
            RVC3), which pass it on to the model optimizer.
        encoding: Color encoding conversion for this input.

    """

    calibration: (
        ImageCalibrationConfig
        | RandomCalibrationConfig
        | LinkCalibrationConfig
    ) = RandomCalibrationConfig()
    scale_values: Annotated[list[float], Field(min_length=1)] | None = None
    mean_values: Annotated[list[float], Field(min_length=1)] | None = None
    frozen_value: list[int | float] | None = None
    encoding: EncodingConfig = EncodingConfig()

    @property
    def encoding_mismatch(self) -> bool:
        """Return whether the ``from`` and ``to`` encodings differ."""
        return self.encoding.from_ != self.encoding.to

    @property
    def is_color_input(self) -> bool:
        """Return whether the input is fed with color image data."""
        return self.encoding.from_ in {Encoding.RGB, Encoding.BGR}

    @property
    def is_raw_input(self) -> bool:
        """Return whether the input is fed with non-image data."""
        return (
            self.encoding.from_ == Encoding.NONE
            and self.encoding.to == Encoding.NONE
        )

    @model_validator(mode="after")
    def _validate_grayscale_inputs(self) -> Self:
        if self.layout is None:
            return self

        if "C" not in self.layout:
            return self

        assert self.shape is not None

        channels = self.shape[self.layout.index("C")]
        if channels == 1:
            logger.info("Detected grayscale input. Setting encoding to GRAY.")
            self.encoding.from_ = self.encoding.to = Encoding.GRAY

        return self

    @model_validator(mode="after")
    def _validate_dynamic_batch_size(self) -> Self:
        if self.shape is not None and self.shape[0] == 0:
            logger.info(
                "Detected dynamic batch size (the first element "
                "of the shape is set to 0). Setting batch size to 1. "
            )
            self.shape[0] = 1
        return self

    @model_validator(mode="before")
    @classmethod
    def _validate_encoding(cls, data: Params) -> Params:
        encoding = data.get("encoding")
        if encoding is None or encoding == {}:
            data["encoding"] = {"from": "RGB", "to": "BGR"}
            return data
        if isinstance(encoding, str):
            data["encoding"] = {"from": encoding, "to": encoding}
        if isinstance(encoding, dict) and (
            ("from" in encoding and encoding["from"] == "GRAY")
            or ("to" in encoding and encoding["to"] == "GRAY")
        ):
            data["encoding"] = {"from": "GRAY", "to": "GRAY"}
        return data

    @model_validator(mode="before")
    @classmethod
    def _random_calibration(cls, data: Params) -> Params:
        if data.get("calibration") in ["random", None]:
            # An empty mapping builds the config from its defaults.
            data["calibration"] = {}
        return data

    @field_validator("scale_values", mode="before")
    @staticmethod
    def _parse_scale_values(value: ParamValue) -> ParamValue:
        """Parse the scale_values from the config."""
        return InputConfig._parse_values("scale", value)

    @field_validator("mean_values", mode="before")
    @staticmethod
    def _parse_mean_values(value: ParamValue) -> ParamValue:
        """Parse the mean_values from the config."""
        return InputConfig._parse_values("mean", value)

    @staticmethod
    def _parse_values(
        values_type: Literal["mean", "scale"], value: ParamValue
    ) -> ParamValue:
        """Resolve named values from the config."""
        if value is None:
            return None
        if isinstance(value, str) and value in NAMED_VALUES:
            return NAMED_VALUES[value][values_type]
        if isinstance(value, float | int):
            return [value, value, value]
        return value

    def requires_onnx_input_modification(
        self, *, reverse_only: bool = False
    ) -> bool:
        """Check whether the ONNX graph must be modified for this input.

        Args:
            reverse_only: If ``True``, only the channel reversal is
                taken into account and the mean and scale values are
                ignored.

        Returns:
            ``True`` if the encodings differ or -- unless
            ``reverse_only`` is set -- if any non-neutral mean or scale
            values are set.

        """
        if self.encoding_mismatch:
            return True
        if reverse_only:
            return False
        return (
            self.mean_values is not None
            and any(v != 0 for v in self.mean_values)
        ) or (
            self.scale_values is not None
            and any(v != 1 for v in self.scale_values)
        )


class PlatformConfig(BaseModelExtraForbid):
    """Base class for the platform-specific sections of a stage.

    Attributes:
        disable_calibration: Whether to skip calibration, and with it
            the quantization of the model.

    """

    disable_calibration: bool = False


class HailoConfig(PlatformConfig):
    """Options of the Hailo conversion.

    Attributes:
        force_onnx_names: Whether to force the ONNX names on the
            produced ``.har`` and ``.hef`` models.
        optimization_level: Optimization level, between ``0`` and ``4``,
            or ``-100`` to disable all optimizations.
        compression_level: Compression level, between ``0`` and ``5``.
        batch_size: Batch size used for the calibration.
        disable_compilation: Whether to stop after the quantization,
            without compiling the model.
        alls: Additional lines of the Hailo model script (``alls``)
            passed to the model optimizer.
        hw_arch: Hailo hardware architecture to compile for.

    """

    force_onnx_names: bool = True
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] = 2
    compression_level: Literal[0, 1, 2, 3, 4, 5] = 2
    batch_size: int = 8
    disable_compilation: bool = False
    alls: list[str] = []
    hw_arch: Literal[
        "hailo8", "hailo8l", "hailo8r", "hailo10h", "hailo15h", "hailo15m"
    ] = "hailo8"


class BlobBaseConfig(PlatformConfig):
    """Options shared by the OpenVINO-based platforms.

    Base class for the RVC2 and RVC3 configurations, which both convert
    the model through OpenVINO into a blob.

    Attributes:
        mo_args: Additional arguments passed to the OpenVINO model
            optimizer. They take precedence over the default ones.
        compile_tool_args: Additional arguments passed to the OpenVINO
            ``compile_tool``. They take precedence over the default
            ones.
        compress_to_fp16: Whether to compress FP32 weights and biases of
            the original model to FP16. All intermediate data is kept in
            the original precision.

    """

    mo_args: list[str] = []
    compile_tool_args: list[str] = []
    compress_to_fp16: bool = True


class RVC2Config(BlobBaseConfig):
    """Options of the RVC2 conversion.

    Attributes:
        number_of_shaves: Number of SHAVE cores the blob is compiled
            for. Forced to ``8`` when ``superblob`` is enabled.
        superblob: Whether to produce a ``.superblob`` file instead of a
            regular ``.blob``.
        n_workers: Number of workers used for the parallel superblob
            compilation. Overrides the automatically determined number.

    """

    number_of_shaves: int = 8
    superblob: bool = True
    n_workers: PositiveInt | None = None

    @model_validator(mode="after")
    def _validate_superblob(self) -> Self:
        if self.superblob and self.number_of_shaves != 8:
            logger.warning("Changing number_of_shaves to 8 for superblob.")
            self.number_of_shaves = 8

        return self


class RVC3Config(BlobBaseConfig):
    """Options of the RVC3 conversion.

    Attributes:
        pot_target_device: Target device of the OpenVINO
            post-training optimization tool.

    """

    pot_target_device: PotDevice = PotDevice.VPU


class QuantizationOverridesItem(BaseModelExtraForbid):
    """Quantization encoding of a single tensor.

    Follows the AIMET encoding specification and is used to override the
    quantization parameters computed during the RVC4 conversion.

    Attributes:
        bitwidth: Number of bits used to represent the tensor.
        is_symmetric: Whether the quantization is symmetric.
        dtype: Whether the tensor is quantized as an integer or a float.
        max: Largest representable value.
        min: Smallest representable value.
        offset: Zero-point offset of the quantization grid.
        scale: Step size of the quantization grid.

    """

    bitwidth: Annotated[int, Literal[4, 8, 16, 32]] | None = None
    is_symmetric: bool | None = None
    dtype: Literal["int", "float"] | None = None
    max: float | None = None
    min: float | None = None
    offset: int | None = None
    scale: float | None = None

    @field_serializer("is_symmetric", when_used="json")
    @staticmethod
    def serialize_is_symmetric(value: bool | None) -> str | None:
        """Serialize ``is_symmetric`` as a string in JSON output.

        Args:
            value: Value of the ``is_symmetric`` field.

        Returns:
            The string representation of the value, or ``None`` if the
            value is ``None``.

        """
        if value is None:
            return None
        return str(value)


class Encodings(BaseModelExtraForbid):
    """Set of quantization encodings overriding the computed ones.

    Attributes:
        activation_encodings: Encodings of the activation tensors, keyed
            by tensor name.
        param_encodings: Encodings of the parameter tensors, keyed by
            tensor name.

    """

    activation_encodings: dict[str, list[QuantizationOverridesItem]]
    param_encodings: dict[str, list[QuantizationOverridesItem]]


class RVC4Config(PlatformConfig):
    """Options of the RVC4 conversion.

    Attributes:
        snpe_onnx_to_dlc_args: Additional arguments passed to SNPE
            ``snpe-onnx-to-dlc``. They take precedence over the default
            ones.
        snpe_dlc_quant_args: Additional arguments passed to SNPE
            ``snpe-dlc-quant``.
        snpe_dlc_graph_prepare_args: Additional arguments passed to SNPE
            ``snpe-dlc-graph-prepare``.
        keep_raw_images: Whether to keep the raw calibration images in
            the intermediate outputs. They can get very large.
        use_per_channel_quantization: Whether to use per-axis-element
            quantization for the weights and biases of the supported
            layer types.
        use_per_row_quantization: Whether to use row-wise quantization
            of the ``MatMul`` and ``FullyConnected`` operations.
        optimization_level: Optimization level of the DLC graph
            preparation. Higher levels take longer but yield a faster
            graph.
        quantization_mode: Pre-defined quantization mode. All modes
            except ``CUSTOM`` override the user-provided SNPE arguments.
        htp_socs: Platforms to pre-compute the DLC graph for.
        encodings: Quantization encodings overriding the computed ones.

    """

    snpe_onnx_to_dlc_args: list[str] = []
    snpe_dlc_quant_args: list[str] = []
    snpe_dlc_graph_prepare_args: list[str] = []
    keep_raw_images: bool = False
    use_per_channel_quantization: bool = True
    use_per_row_quantization: bool = False
    optimization_level: Literal[1, 2, 3] = 2
    quantization_mode: QuantizationMode = QuantizationMode.INT8_STD
    htp_socs: list[
        Literal["sm8350", "sm8450", "sm8550", "sm8650", "qcs6490", "qcs8550"]
    ] = ["sm8550"]
    encodings: Encodings | None = None

    @model_validator(mode="after")
    def validate_quantization_overrides(self) -> Self:
        """Move ``--quantization_overrides`` into `Encodings`.

        The flag and the path following it are removed from
        ``snpe_onnx_to_dlc_args`` and the referenced JSON file is parsed
        into the ``encodings`` field.

        Returns:
            The validated model.

        Raises:
            ValueError: If both ``--quantization_overrides`` and
                ``encodings`` are specified.

        """
        if "--quantization_overrides" in self.snpe_onnx_to_dlc_args:
            if self.encodings:
                raise ValueError(
                    "Cannot specify both `--quantization_overrides`"
                    "in `rvc4.snpe_onnx_to_dlc_args` and "
                    "`rvc4.encodings` at the same time."
                )
            qo_index = self.snpe_onnx_to_dlc_args.index(
                "--quantization_overrides"
            )
            self.snpe_onnx_to_dlc_args.pop(qo_index)
            encodings_json = self.snpe_onnx_to_dlc_args.pop(qo_index)
            with open(encodings_json) as f:
                self.encodings = parse_encodings(f.read())
        return self

    @field_validator("encodings", mode="before")
    @staticmethod
    def validate_encodings(value: ParamValue | Encodings) -> Encodings | None:
        """Parse the quantization encodings from the config.

        Args:
            value: Either ``None``, a JSON string, a path to a JSON
                file, or an already parsed mapping.

        Returns:
            The parsed encodings, or ``None`` if no value was given.

        """
        if value is None:
            return None

        if isinstance(value, str):
            if value.lstrip().startswith("{"):
                return parse_encodings(value)
            value_path = resolve_path(value, MISC_DIR)
            return parse_encodings(value_path.read_text())

        return parse_encodings(value)

    @model_validator(mode="after")
    def _validate_fp16(self) -> Self:
        if self.quantization_mode != QuantizationMode.FP16_STD:
            return self
        self.disable_calibration = True
        return self


class ONNXOptimizationsConfig(BaseModelExtraForbid):
    """Switches of the individual ONNX graph optimizations.

    The optimizations are applied to the ONNX model before it is handed
    over to the platform-specific conversion. All of them are enabled by
    default.

    Attributes:
        fuse_add_mul_to_bn: Whether to fuse ``Add``/``Sub`` and ``Mul``
            nodes following a ``Conv`` node into a
            ``BatchNormalization`` node.
        fuse_comb_add_mul_to_conv: Whether to fuse combinations of
            ``Add`` and ``Mul`` nodes preceding a ``Conv`` node into the
            ``Conv`` node.
        fuse_single_add_mul_to_conv: Whether to fuse a single ``Add`` or
            ``Mul`` node preceding a ``Conv`` node into the ``Conv``
            node.
        fuse_split_concat_to_conv: Whether to fuse ``Split`` and
            ``Concat`` nodes preceding a ``Conv`` node into the ``Conv``
            node.
        substitute_sub_with_add: Whether to replace ``Sub`` nodes with
            ``Add`` and ``Neg``.
        substitute_div_with_mul: Whether to replace ``Div`` nodes with
            ``Mul`` and ``Reciprocal``.

    """

    fuse_add_mul_to_bn: bool = True
    fuse_comb_add_mul_to_conv: bool = True
    fuse_single_add_mul_to_conv: bool = True
    fuse_split_concat_to_conv: bool = True
    substitute_sub_with_add: bool = True
    substitute_div_with_mul: bool = True

    def all_disabled(self) -> bool:
        """Check whether every optimization is switched off."""
        return not any(
            [
                self.fuse_add_mul_to_bn,
                self.fuse_comb_add_mul_to_conv,
                self.fuse_single_add_mul_to_conv,
                self.fuse_split_concat_to_conv,
                self.substitute_sub_with_add,
                self.substitute_div_with_mul,
            ]
        )


class SingleStageConfig(BaseModelExtraForbid):
    """Description of a single stage of the conversion.

    Holds everything needed to convert one model: the model itself, its
    inputs and outputs, and the options of every supported platform.
    Most of the fields are filled in from the metadata of the input
    model when they are not given explicitly.

    Attributes:
        input_model: Path to the model to convert. Downloaded to the
            local models directory during validation.
        input_bin: Path to the ``.bin`` file of an OpenVINO IR model.
            Derived from ``input_model`` for IR models.
        input_file_type: Type of the input model, derived from the
            suffix of ``input_model``.
        inputs: Descriptions of the model inputs.
        outputs: Descriptions of the model outputs.
        keep_intermediate_outputs: Whether to keep the files created
            during the conversion.
        onnx_simplification: ONNX simplification method to use, or
            ``False`` to disable the simplification.
        onnx_optimizations: Switches of the ONNX graph optimizations.
        output_remote_url: Remote URL to upload the converted model to.
        intermediate_outputs_remote_url: Remote URL to upload the
            intermediate outputs to.
        put_file_plugin: Name of the registered plugin used for the
            uploads.
        hailo: Options of the Hailo conversion.
        rvc2: Options of the RVC2 conversion.
        rvc3: Options of the RVC3 conversion.
        rvc4: Options of the RVC4 conversion.

    """

    input_model: Path
    input_bin: Path | None = None
    input_file_type: InputFileType

    inputs: Annotated[list[InputConfig], Field(min_length=1)] = []
    outputs: Annotated[list[OutputConfig], Field(min_length=1)] = []

    keep_intermediate_outputs: bool = True
    onnx_simplification: Literal["onnxsim", "onnxslim", False] = "onnxsim"
    onnx_optimizations: ONNXOptimizationsConfig = ONNXOptimizationsConfig()
    output_remote_url: str | None = None
    intermediate_outputs_remote_url: str | None = None
    put_file_plugin: str | None = None

    hailo: HailoConfig = HailoConfig()
    rvc2: RVC2Config = RVC2Config()
    rvc3: RVC3Config = RVC3Config()
    rvc4: RVC4Config = RVC4Config()

    @model_validator(mode="before")
    @classmethod
    def validate_onnx_simplification(cls, data: Params) -> Params:
        """Translate the deprecated ``disable_onnx_simplification``.

        .. deprecated:: 0.5.6
            Use ``onnx_simplification: false`` instead.

        Args:
            data: Raw stage configuration.

        Returns:
            The configuration with ``onnx_simplification`` set to
            ``False`` if the deprecated option was enabled.

        Raises:
            ValueError: If ``disable_onnx_simplification`` is enabled
                and ``onnx_simplification`` is specified as well.

        """
        if data.pop("disable_onnx_simplification", False):
            if "onnx_simplification" in data:
                raise ValueError(
                    "Cannot specify both `disable_onnx_simplification` "
                    "and `onnx_simplification`."
                )

            warnings.warn(
                "`disable_onnx_simplification` is deprecated. Please use "
                "`onnx_simplification` set to `False` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data["onnx_simplification"] = False
        return data

    def get_platform_config(self, platform: Platform) -> PlatformConfig:
        """Return the platform configuration for the given platform."""
        if platform == Platform.HAILO:
            return self.hailo
        if platform == Platform.RVC2:
            return self.rvc2
        if platform == Platform.RVC3:
            return self.rvc3
        if platform == Platform.RVC4:  # pragma: no branch
            return self.rvc4

    @model_validator(mode="before")
    @classmethod
    def validate_onnx_optimizations(cls, data: Params) -> Params:
        """Expand the shorthand values of ``onnx_optimizations``.

        Args:
            data: Raw stage configuration.

        Returns:
            The configuration with ``onnx_optimizations`` replaced by an
            `ONNXOptimizationsConfig` with all optimizations either
            enabled or disabled, if a shorthand value was used.

        """
        if "onnx_optimizations" not in data:
            return data
        optimizations = data["onnx_optimizations"]
        if optimizations in ["all", True]:
            data["onnx_optimizations"] = {}
        elif optimizations in ["none", None, False]:
            data["onnx_optimizations"] = dict.fromkeys(
                ONNXOptimizationsConfig.model_fields, False
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_disable_onnx_optimizations(cls, data: Params) -> Params:
        """Translate the deprecated ``disable_onnx_optimizations``.

        .. deprecated:: 0.5.6
            Use ``onnx_optimizations: false`` instead.

        Args:
            data: Raw stage configuration.

        Returns:
            The configuration with all ONNX optimizations disabled if
            the deprecated option was enabled.

        Raises:
            ValueError: If ``disable_onnx_optimizations`` is enabled
                and ``onnx_optimizations`` is specified as well.

        """
        if "disable_onnx_optimizations" not in data:
            return data
        if data.pop("disable_onnx_optimizations", False):
            warnings.warn(
                "`disable_onnx_optimizations` is deprecated. Please use "
                "`onnx_optimizations: false` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "onnx_optimizations" in data:
                raise ValueError(
                    "Cannot specify both `disable_onnx_optimizations` and "
                    "`onnx_optimizations` at the same time."
                )
            data["onnx_optimizations"] = dict.fromkeys(
                ONNXOptimizationsConfig.model_fields, False
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_model(cls, data: Params) -> Params:
        mean_values = data.pop("mean_values", None)
        scale_values = data.pop("scale_values", None)
        encoding = data.pop("encoding", {})
        data_type = data.pop("data_type", None)
        shape = data.pop("shape", None)
        layout = data.pop("layout", None)
        top_level_calibration = data.pop("calibration", {})

        model_path = Path(_as_path_type(data["input_model"], "input_model"))

        input_file_type = InputFileType.from_path(model_path)
        data["input_file_type"] = input_file_type.value
        if input_file_type == InputFileType.PYTORCH:
            logger.info(
                "Detected PyTorch model. Only YOLO models are supported."
            )
            raw_shape = data.pop("yolo_input_shape", [640, 640])
            if isinstance(raw_shape, str):
                input_shape = (
                    [int(size) for size in raw_shape.split(" ")]
                    if " " in raw_shape
                    else [int(raw_shape)] * 2
                )
            else:
                logger.warning(
                    "yolo_input_shape is not provided. Using default shape [640, 640]."
                )
                input_shape = _as_shape(raw_shape, "yolo_input_shape")
            input_shapes = {"images": input_shape[::-1]}
            input_dtypes = {"images": DataType.FLOAT32}
            output_shapes = {"dummy": [0]}
            output_dtypes = {"dummy": DataType.FLOAT32}

            metadata = Metadata(
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
            )
        else:
            metadata = get_metadata(model_path)

        inputs: list[Params] = _as_entries(data.get("inputs"), "inputs")
        if not inputs:
            inputs = [{"name": name} for name in metadata.input_shapes]
        outputs: list[Params] = _as_entries(data.get("outputs"), "outputs")
        if not outputs:
            outputs = [{"name": name} for name in metadata.output_shapes]

        for inp in inputs:
            if "name" not in inp:
                raise ValueError(
                    f"Unable to determine name for input: `{inp}`."
                )
            inp_name = str(inp["name"])
            if inp_name in metadata.input_shapes:
                onnx_shape, onnx_dtype = (
                    metadata.input_shapes[inp_name],
                    metadata.input_dtypes[inp_name],
                )
            else:
                onnx_shape, onnx_dtype = _get_onnx_inter_info(
                    model_path, inp_name
                )
                logger.warning(
                    f"Input `{inp_name}` is not present in inputs of the ONNX model. "
                    f"Assuming it is an intermediate node."
                )
            inp["shape"] = inp.get("shape") or shape or onnx_shape
            inp["layout"] = inp.get("layout") or layout
            inp["data_type"] = (
                inp.get("data_type") or data_type or _dtype_name(onnx_dtype)
            )
            inp["encoding"] = inp.get("encoding") or encoding
            inp["mean_values"] = (
                inp.get("mean_values")
                if inp.get("mean_values") is not None
                else mean_values
            )
            inp["scale_values"] = (
                inp.get("scale_values")
                if inp.get("scale_values") is not None
                else scale_values
            )

            inp_calibration = inp.get("calibration", {})
            if not inp_calibration and not top_level_calibration:
                inp["calibration"] = None
            elif top_level_calibration == "random":
                inp["calibration"] = "random"
            else:
                inp["calibration"] = {
                    **_as_dict(top_level_calibration, "calibration"),
                    **_as_dict(inp_calibration, "calibration"),
                }

        for out in outputs:
            out_name = str(out["name"])
            if (
                out_name not in metadata.output_shapes
                and out.get("data_type") is None
                and out.get("shape") is None
            ):
                onnx_shape, onnx_dtype = _get_onnx_inter_info(
                    model_path, out_name
                )
            elif out_name in metadata.output_shapes:
                onnx_shape, onnx_dtype = (
                    metadata.output_shapes[out_name],
                    metadata.output_dtypes[out_name],
                )
            else:
                onnx_shape, onnx_dtype = None, None
            out["shape"] = out.get("shape") or onnx_shape
            out["data_type"] = out.get("data_type") or _dtype_name(onnx_dtype)

        data["inputs"] = inputs
        data["outputs"] = outputs

        disable_calibration = data.pop("disable_calibration", None)
        if disable_calibration is None:
            return data

        for platform in ["hailo", "rvc2", "rvc3", "rvc4"]:
            platform_data = _as_dict(data.setdefault(platform, {}), platform)
            platform_data["disable_calibration"] = disable_calibration
        return data

    @model_validator(mode="before")
    @classmethod
    def _download_input_model(cls, value: Params) -> Params:
        if "input_model" not in value:
            raise ValueError("`input_model` must be provided.")
        input_model = _as_path_type(value["input_model"], "input_model")
        input_file_type = InputFileType.from_path(input_model)
        if input_file_type == InputFileType.IR:
            bin_path, xml_path = _extract_bin_xml_from_ir(input_model)
            value["input_bin"] = str(bin_path)
            value["input_model"] = str(xml_path)
        else:
            value["input_model"] = str(
                resolve_path(str(input_model), MODELS_DIR)
            )
        return value


# TODO: Output remote url
class Config(LuxonisConfig):
    """Top-level configuration of a conversion.

    A single-stage configuration is accepted as well: the stage fields
    can be given at the top level and are wrapped into a single stage
    during validation, so the shorthand

    .. code-block:: yaml

        input_model: models/yolov6n.onnx
        encoding:
          from: RGB
          to: BGR
        calibration:
          path: models/coco128

    is equivalent to the same fields nested under a single entry of
    ``stages``. See ``configs/defaults.yaml`` for every field with its
    default.

    Attributes:
        stages: Configurations of the individual stages, keyed by stage
            name.
        name: Name of the model. Defaults to the stage names joined by
            dashes, or to the stem of the input model for a single
            unnamed stage.
        rich_logging: Whether to use rich formatting for the log
            messages.

    """

    stages: Annotated[dict[str, SingleStageConfig], Field(min_length=1)]
    name: str
    rich_logging: bool = True

    def get_stage_config(self, stage: str | None) -> SingleStageConfig:
        """Return the configuration of the given stage.

        Args:
            stage: Name of the stage. Can be ``None`` only if the
                configuration has a single stage.

        Returns:
            The configuration of the requested stage.

        Raises:
            ValueError: If no stage name is given and the configuration
                has more than one stage.
            KeyError: If no stage of the given name exists.

        """
        if stage is None:
            if len(self.stages) == 1:
                return next(iter(self.stages.values()))
            raise ValueError("Multiple stages found. Please specify a stage.")
        return self.stages[stage]

    @model_validator(mode="before")
    @classmethod
    def _validate_name(cls, data: Params) -> Params:
        if data.get("name") is None:
            stages = _as_dict(data["stages"], "stages")
            data["name"] = "-".join(stages.keys())
        return data

    @model_validator(mode="before")
    @classmethod
    def _validate_stages(cls, data: Params) -> Params:
        if "stages" not in data:
            name = data.pop("name", "default_stage")
            if not isinstance(name, str):
                raise TypeError("`name` must be a string.")
            rich_logging = data.pop("rich_logging", True)
            return {
                "name": name,
                "rich_logging": rich_logging,
                "stages": {name: data},
            }

        extra: Params = {}
        for key in list(data.keys()):
            if key not in cls.model_fields:
                extra[key] = data.pop(key)
        stages = _as_dict(data["stages"], "stages")
        for stage_name, stage in stages.items():
            stage_data = _as_dict(stage, f"stages.{stage_name}")
            for key, value in extra.items():
                if key not in stage_data:
                    stage_data[key] = value
        return data

    @model_validator(mode="after")
    def _validate_single_stage_name(self) -> Self:
        """Change the default 'default_stage' name to the name of the
        input model.
        """
        if len(self.stages) == 1 and "default_stage" in self.stages:
            stage = next(iter(self.stages.values()))
            model_name = stage.input_model.stem
            self.stages = {model_name: stage}
            self.name = model_name
        return self


def _dtype_name(dtype: DataType | None) -> str | None:
    return None if dtype is None else dtype.value


def _as_dict(value: ParamValue, name: str) -> Params:
    if not isinstance(value, dict):
        raise TypeError(f"`{name}` must be a dictionary.")
    return value


def _as_entries(value: ParamValue, name: str) -> list[Params]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"`{name}` must be a list of dictionaries.")
    entries = [entry for entry in value if isinstance(entry, dict)]
    if len(entries) != len(value):
        raise TypeError(f"`{name}` must be a list of dictionaries.")
    return entries


def _as_shape(value: ParamValue, name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"`{name}` must be a list of integers.")
    dims = [dim for dim in value if isinstance(dim, int)]
    if len(dims) != len(value):
        raise TypeError(f"`{name}` must be a list of integers.")
    return dims


def _as_path_type(value: ParamValue, name: str) -> PathType:
    """Narrow a raw configuration value to something ``Path`` accepts.

    Returned as it arrived rather than as a ``Path``: a remote location
    is carried in these fields as a string, and ``Path`` would fold the
    ``//`` of its protocol away.

    Args:
        value: Raw value taken from the configuration.
        name: Name of the field, used in the error message.

    Returns:
        The value unchanged.

    Raises:
        TypeError: If the value is neither a string nor a path.

    """
    if not isinstance(value, PathType):
        raise TypeError(f"`{name}` must be a string or a path.")
    return value


def _extract_bin_xml_from_ir(ir_path: ParamValue | Path) -> tuple[Path, Path]:
    """Extract the corresponding second path from a single IR path.

    The base file name must match between the ``.bin`` and the ``.xml``
    file. Otherwise, an error is raised.

    Args:
        ir_path: The path of either the ``.bin`` or the ``.xml`` file.
            It arrives unvalidated, straight out of the configuration,
            so it is only a path once the check below has run.

    Returns:
        The ``.bin`` path and the ``.xml`` path.

    Raises:
        TypeError: If ``ir_path`` is neither a string nor a path.
        ValueError: If the path has neither the ``.bin`` nor the
            ``.xml`` extension.

    """
    if not isinstance(ir_path, PathType):
        raise TypeError("`input_path` must be str or Path.")
    path = Path(ir_path)

    if path.suffix == ".bin":
        bin_path = str(path)
        xml_path = str(path.with_suffix(".xml"))
    elif path.suffix == ".xml":
        xml_path = str(path)
        bin_path = str(path.with_suffix(".bin"))
    else:
        raise ValueError(
            "`ir_path` is invalid: does not have .bin or .xml extension."
        )

    # fix any remote path corruption from pathlib
    bin_path = bin_path.replace(":/", "://")
    xml_path = xml_path.replace(":/", "://")

    try:
        resolved_bin = resolve_path(bin_path, MODELS_DIR)
    except Exception as e:
        raise ValueError(
            f"`bin_path` {bin_path} was not found. "
            "Please ensure that your xml and bin file have matching file basenames "
            "and are located in the same directory."
        ) from e
    try:
        resolved_xml = resolve_path(xml_path, MODELS_DIR)
    except Exception as e:
        raise ValueError(
            f"`xml_path` {xml_path} was not found. "
            "Please ensure that your xml and bin file have matching file basenames and "
            "are located in the same directory."
        ) from e

    return resolved_bin, resolved_xml


def _get_onnx_node_info(
    model_path: Path, node_name: str
) -> tuple[list[int], DataType]:
    onnx_model = onnx.load(str(model_path))
    graph = onnx_model.graph

    node = next((n for n in graph.node if n.name == node_name), None)
    if node is None:
        raise NameError(f"Node '{node_name}' not found in the ONNX model.")

    output_value_info = next(
        (info for info in graph.value_info if info.name == node.output[0]),
        None,
    )

    if output_value_info is None:
        raise ValueError(
            f"Output value info for node '{node_name}' not found."
        )

    shape = _get_static_onnx_shape(
        output_value_info.type.tensor_type, f"node '{node_name}'"
    )
    data_type = output_value_info.type.tensor_type.elem_type

    return shape, DataType.from_onnx_dtype(data_type)


def _get_onnx_tensor_info(
    model_path: PathType, tensor_name: str
) -> tuple[list[int], DataType]:
    model = onnx.load(str(model_path))

    def extract_tensor_info(
        tensor_type: TypeProto.Tensor,
    ) -> tuple[list[int], DataType]:
        shape = _get_static_onnx_shape(tensor_type, f"tensor '{tensor_name}'")
        return shape, DataType.from_onnx_dtype(tensor_type.elem_type)

    for tensor in chain(model.graph.input, model.graph.output):
        if tensor.name == tensor_name:
            return extract_tensor_info(tensor.type.tensor_type)

    for node in model.graph.node:
        for tensor in chain(node.input, node.output):
            if tensor == tensor_name:
                for value_info in model.graph.value_info:
                    if value_info.name == tensor_name:
                        return extract_tensor_info(value_info.type.tensor_type)
                raise ValueError(
                    f"Tensor '{tensor_name}' does not have shape/type information."
                )

    raise NameError(f"Tensor '{tensor_name}' not found in the ONNX model.")


def _get_static_onnx_shape(
    tensor_type: TypeProto.Tensor, tensor_name: str
) -> list[int]:
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            raise ValueError(
                "Dynamic shapes are not supported. "
                f"Shape of {tensor_name} is {[d.dim_value for d in tensor_type.shape.dim]}."
            )
    return shape


def _get_onnx_inter_info(
    model_path: Path, name: str
) -> tuple[list[int] | None, DataType | None]:
    try:
        logger.info(
            f"Attempting to find shape and data type for tensor '{name}'."
        )
        shape, data_type = _get_onnx_tensor_info(model_path, name)
    except (NameError, ValueError) as e:
        logger.warning(str(e))
        logger.info(
            f"Attempting to find shape and data type for node '{name}'."
        )
        try:
            shape, data_type = _get_onnx_node_info(model_path, name)
        except (NameError, ValueError) as e:
            logger.warning(str(e))
            shape, data_type = None, None
    if shape is None or data_type is None:
        logger.warning(
            f"Tensor or node '{name}' not found or does not have shape/type information. "
            "Proceeding without shape and data type information."
        )
    else:
        logger.info(
            f"Found shape and data type for '{name}': {shape}, {data_type.name}"
        )
    return shape, data_type


def generate_renamed_onnx(
    onnx_path: PathType,
    rename_dict: dict[str, str],
    output_path: PathType,
) -> None:
    """Save a copy of an ONNX model with renamed node connections.

    Every node input and output whose name appears in ``rename_dict`` is
    replaced by the new name. The model is saved with external data if
    the original model used it.

    Args:
        onnx_path: Path to the ONNX model to rename.
        rename_dict: Mapping from the old tensor names to the new ones.
        output_path: Path to save the renamed model to.

    """
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    model = onnx.load(str(onnx_path))
    model_has_external_data = has_external_data(onnx_path)

    for node in model.graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in rename_dict:
                node.input[i] = rename_dict[input_name]

        for i, output_name in enumerate(node.output):
            if output_name in rename_dict:
                node.output[i] = rename_dict[output_name]

    save_onnx_model(
        model,
        output_path,
        save_as_external_data=model_has_external_data,
        location=f"{output_path.name}_data",
    )
