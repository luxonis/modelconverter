"""RVC4 conversion backend.

Implements the RVC4 side of modelconverter: exporting an ONNX or
TFLite model to a ``.dlc`` file with the SNPE tools
(``snpe-onnx-to-dlc`` or ``snpe-tflite-to-dlc``, then
``snpe-dlc-quant`` and ``snpe-dlc-graph-prepare``), plus inference,
benchmarking, analysis and visualization of the converted model. The
classes in this package are picked by platform through the getters in
`modelconverter.platforms`.

.. note::
    The export runs inside the RVC4 Docker image, which ships the
    SNPE tools. The host-run ``analyze`` command calls
    ``snpe-dlc-info`` and ``snpe-diagview`` directly, so it needs
    them on the host ``PATH``. Benchmarking and analysis additionally
    need a real RVC4 device, reachable over ADB or SSH.
"""
