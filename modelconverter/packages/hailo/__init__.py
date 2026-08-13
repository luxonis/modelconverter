"""Conversion support for Hailo devices.

Holds the exporter, which compiles an ONNX or TFLite model into the
Hailo ``.hef`` format, and the inferer, which runs a translated
``.har`` model in the SDK's quantized inference context.

.. note::
    Both are built on ``hailo_sdk_client``, which is installed in the
    Hailo Docker image and not on the host, so they can only be
    imported inside that image.
"""
