"""Inference with models converted for the RVC3 platform.

An RVC3 conversion goes through the same OpenVINO IR as an RVC2 one, and
inference is run on that IR, so the RVC2 inferer is reused unchanged and
only re-exported under an RVC3 name.
"""

from modelconverter.platforms.rvc2.inferer import RVC2Inferer as RVC3Inferer

__all__ = ["RVC3Inferer"]
