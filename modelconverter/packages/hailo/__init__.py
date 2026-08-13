"""Conversion support for Hailo devices.

Holds the exporter, which compiles an ONNX or TFLite model into the
Hailo ``.hef`` format, and the inferer, which runs a translated
``.har`` model in the SDK's quantized inference context. Both are
built on the ``hailo_sdk_client`` toolchain and are meant to run
inside the Hailo Docker image, where that toolchain is installed.
"""
