from pathlib import Path
from typing import Literal

from luxonis_ml.typing import Kwargs, PathType

from modelconverter.utils.types import PotDevice, Target

from .__main__ import convert as cli_convert


class convert:
    """Namespace for all conversion methods.

    Specific conversion functions for individual targets are
    available as static methods under the names `RVC2`, `RVC3`, `RVC4`, and `Hailo`.
    """

    __call__ = staticmethod(cli_convert)

    @staticmethod
    def RVC2(
        path: PathType,
        mo_args: list[str] | None = None,
        compile_tool_args: list[str] | None = None,
        compress_to_fp16: bool = True,
        number_of_shaves: int = 8,
        superblob: bool = True,
        opts: Kwargs | list[str] | None = None,
        **hub_kwargs,
    ) -> Path:
        """Convert a model to RVC2 format.

        Parameters
        ----------
        path : PathType
            Path to the model file to be converted.
        mo_args : list[str] | None, optional
            Additional arguments for the Model Optimizer (MO).
        compile_tool_args : list[str] | None, optional
            Additional arguments for the compile tool.
        compress_to_fp16 : bool, default True
            Whether to compress the model's weights to FP16.
        number_of_shaves : int, default 8
            Number of shaves to use for the conversion.
        superblob : bool, default True
            Whether to create a superblob for the model.
        opts : dict[str, Any] | list[str] | None, optional
            Additional options for the conversion. Can be used
            to override configuration values.
        **hub_kwargs
            Additional keyword arguments to be passed to the
            online conversion.
        """
        return cli_convert(
            Target.RVC2,
            _combine_opts(
                Target.RVC2,
                {
                    "mo_args": mo_args or [],
                    "compile_tool_args": compile_tool_args or [],
                    "compress_to_fp16": compress_to_fp16,
                    "number_of_shaves": number_of_shaves,
                    "superblob": superblob,
                },
                opts,
            ),
            path=str(path),
            **hub_kwargs,
        )

    @staticmethod
    def RVC3(
        path: PathType,
        mo_args: list[str] | None = None,
        compile_tool_args: list[str] | None = None,
        compress_to_fp16: bool = True,
        pot_target_device: PotDevice | Literal["VPU", "ANY"] = PotDevice.VPU,
        opts: Kwargs | list[str] | None = None,
        **hub_kwargs,
    ) -> Path:
        """Convert a model to RVC3 format.

        Parameters
        ----------
        path : PathType
            Path to the model file to be converted.
        mo_args : list[str] | None, optional
            Additional arguments for the Model Optimizer (MO).
        compile_tool_args : list[str] | None, optional
            Additional arguments for the compile tool.
        compress_to_fp16 : bool, default True
            Whether to compress the model's weights to FP16.
        pot_target_device : PotDevice | Literal["VPU", "ANY"], default PotDevice.VPU
            Target device for POT quantization.
        opts : dict[str, Any] | list[str] | None, optional
            Additional options for the conversion. Can be used
            to override configuration values.
        **hub_kwargs
            Additional keyword arguments to be passed to the
            online conversion.
        """
        if not isinstance(pot_target_device, PotDevice):
            pot_target_device = PotDevice(pot_target_device)
        return cli_convert(
            Target.RVC3,
            _combine_opts(
                Target.RVC3,
                {
                    "mo_args": mo_args or [],
                    "compile_tool_args": compile_tool_args or [],
                    "compress_to_fp16": compress_to_fp16,
                    "pot_target_device": pot_target_device.value,
                },
                opts,
            ),
            path=str(path),
            **hub_kwargs,
        )

    @staticmethod
    def RVC4(
        path: PathType,
        snpe_onnx_to_dlc_args: list[str] | None = None,
        snpe_dlc_quant_args: list[str] | None = None,
        snpe_dlc_graph_prepare_args: list[str] | None = None,
        use_per_channel_quantization: bool = True,
        use_per_row_quantization: bool = False,
        htp_socs: list[
            Literal[
                "sm8350", "sm8450", "sm8550", "sm8650", "qcs6490", "qcs8550"
            ]
        ]
        | None = None,
        opts: Kwargs | list[str] | None = None,
        **hub_kwargs,
    ) -> Path:
        """Convert a model to RVC4 format.

        Parameters
        ----------
        path : PathType
            Path to the model file to be converted.
        snpe_onnx_to_dlc_args : list[str] | None, optional
            Additional arguments for the SNPE ONNX to DLC conversion.
        snpe_dlc_quant_args : list[str] | None, optional
            Additional arguments for the SNPE DLC quantization.
        snpe_dlc_graph_prepare_args : list[str] | None, optional
            Additional arguments for the SNPE DLC graph preparation.
        use_per_channel_quantization : bool, default True
            Whether to use per-channel quantization.
        use_per_row_quantization : bool, default False
            Whether to use per-row quantization.
        htp_socs : list[str] | None, optional
            List of HTP SoCs for the final DLC graph.
        opts : dict[str, Any] | list[str] | None, optional
            Additional options for the conversion. Can be used
            to override configuration values.
        **hub_kwargs
            Additional keyword arguments to be passed to the
            online conversion.
        """
        htp_socs = htp_socs or ["sm8550"]
        return cli_convert(
            Target.RVC4,
            _combine_opts(
                Target.RVC4,
                {
                    "snpe_onnx_to_dlc_args": snpe_onnx_to_dlc_args or [],
                    "snpe_dlc_quant_args": snpe_dlc_quant_args or [],
                    "snpe_dlc_graph_prepare_args": snpe_dlc_graph_prepare_args
                    or [],
                    "usu_per_channel_quantization": use_per_channel_quantization
                    or [],
                    "use_per_row_quantization": use_per_row_quantization,
                    "htp_socs": htp_socs,
                },
                opts,
            ),
            path=str(path),
            **hub_kwargs,
        )

    @staticmethod
    def Hailo(
        path: PathType,
        optimization_level: Literal[-100, 0, 1, 2, 3, 4] = 2,
        compression_level: Literal[0, 1, 2, 3, 4, 5] = 2,
        batch_size: int = 8,
        alls: list[str] | None = None,
        opts: Kwargs | list[str] | None = None,
        **hub_kwargs,
    ) -> Path:
        """Convert a model to Hailo format.

        Parameters
        ----------
        path : PathType
            Path to the model file to be converted.
        optimization_level : int, default 2
            Optimization level for the conversion.
        compression_level : int, default 2
            Compression level for the conversion.
        batch_size : int, default 8
            Batch size for the conversion.
        alls : list[str] | None, optional
            List of `alls` parameters for the conversion.
        opts : dict[str, Any] | list[str] | None, optional
            Additional options for the conversion. Can be used
            to override configuration values.
        **hub_kwargs
            Additional keyword arguments to be passed to the
            online conversion.
        """
        return cli_convert(
            Target.HAILO,
            _combine_opts(
                Target.HAILO,
                {
                    "optimization_level": optimization_level,
                    "compression_level": compression_level,
                    "batch_size": batch_size,
                    "alls": alls or [],
                },
                opts,
            ),
            path=str(path),
            **hub_kwargs,
        )


def _combine_opts(
    target: Target, target_kwargs: Kwargs, opts: list[str] | Kwargs | None
) -> list[str]:
    opts = opts or []
    if isinstance(opts, dict):
        opts_list = []
        for key, value in opts.items():
            opts_list.extend([key, value])
    else:
        opts_list = opts

    for key, value in target_kwargs.items():
        opts_list.extend([f"{target.value}.{key}", value])

    return opts_list
