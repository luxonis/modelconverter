"""CPU inference for RVC3's quantized OpenVINO IR."""

from modelconverter.platforms.rvc2.inferer import RVC2Inferer


class RVC3Inferer(RVC2Inferer):
    """Evaluate quantization in floating point to avoid CPU INT8 saturation.

    VPU-targeted weights use all eight bits. OpenVINO's CPU INT8 kernels can
    saturate on AVX2/non-VNNI hosts, corrupting accuracy checks and linked
    calibration. FakeQuantize still rounds and clips to the exported ranges;
    only CPU arithmetic changes. The device artifact is untouched.
    """

    # Set this on the plugin: load_network(config=...) cannot override the
    # plugin's enabled-by-default low-precision transformations in OV 2022.3.
    _cpu_config = {"LP_TRANSFORMS_MODE": "NO"}
