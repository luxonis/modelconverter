"""Conversion backend for the RVC2 platform.

Holds the exporter, inferer and benchmark for RVC2. The exporter
accepts an ONNX model, an OpenVINO IR pair or a TFLite model; a TFLite
model is converted to ONNX first, which the exporter itself logs as
experimental. The conversion then goes through OpenVINO: its model
optimizer turns the ONNX model into an OpenVINO IR and ``compile_tool``
compiles that into a ``.blob``. The code in this package runs inside
the RVC2 Docker image.
"""
