"""Conversion backend for the RVC2 platform.

Holds the exporter, inferer and benchmark for RVC2. The exporter
accepts an ONNX model, an OpenVINO IR pair or a TFLite model; a TFLite
model is converted to ONNX first, which the exporter itself logs as
experimental. The conversion then goes through OpenVINO: its model
optimizer turns the ONNX model into an OpenVINO IR and ``compile_tool``
compiles that into a ``.blob`` -- by default one blob per SHAVE count,
packed into a single ``.superblob``.

.. note::
    These modules import the OpenVINO toolchain, which exists only
    inside the RVC2 Docker image, so they do not import on the host.
"""
