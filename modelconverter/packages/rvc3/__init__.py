"""Conversion, inference and benchmarking for the RVC3 platform.

RVC3 shares the OpenVINO toolchain with RVC2, so `RVC3Exporter` builds
on the RVC2 exporter and adds the RVC3-specific quantization and
compilation steps, while the inferer is the RVC2 one under a different
name. It accepts the same inputs as RVC2: an ONNX model, an OpenVINO
IR pair, or a TFLite model, which is converted to ONNX first. The
modules here run inside the RVC3 Docker image, which is where the
OpenVINO tools they call live.
"""
