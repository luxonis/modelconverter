from pathlib import Path
from typing import Literal

from luxonis_ml.typing import Kwargs, PathType

from modelconverter.utils.types import PotDevice, Target

from .__main__ import convert as cli_convert


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


def RVC3(
    path: PathType,
    mo_args: list[str] | None = None,
    compile_tool_args: list[str] | None = None,
    compress_to_fp16: bool = True,
    pot_target_device: PotDevice | Literal["VPU", "ANY"] = PotDevice.VPU,
    opts: Kwargs | list[str] | None = None,
    **hub_kwargs,
) -> Path:
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


def RVC4(
    path: PathType,
    snpe_onnx_to_dlc_args: list[str] | None = None,
    snpe_dlc_quant_args: list[str] | None = None,
    snpe_dlc_graph_prepare_args: list[str] | None = None,
    usu_per_channel_quantization: bool = True,
    use_per_row_quantization: bool = False,
    htp_socs: list[
        Literal["sm8350", "sm8450", "sm8550", "sm8650", "qcs6490", "qcs8550"]
    ]
    | None = None,
    opts: Kwargs | list[str] | None = None,
    **hub_kwargs,
) -> Path:
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
                "usu_per_channel_quantization": usu_per_channel_quantization
                or [],
                "use_per_row_quantization": use_per_row_quantization,
                "htp_socs": htp_socs,
            },
            opts,
        ),
        path=str(path),
        **hub_kwargs,
    )


def Hailo(
    path: PathType,
    optimization_level: Literal[-100, 0, 1, 2, 3, 4] = 2,
    compression_level: Literal[0, 1, 2, 3, 4, 5] = 2,
    batch_size: int = 8,
    alls: list[str] | None = None,
    opts: Kwargs | list[str] | None = None,
    **hub_kwargs,
) -> Path:
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
