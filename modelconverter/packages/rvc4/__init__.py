"""RVC4 conversion backend.

Implements the RVC4 side of modelconverter: exporting an ONNX or
TFLite model to a ``.dlc`` file with the SNPE tools
(``snpe-onnx-to-dlc`` or ``snpe-tflite-to-dlc``, then
``snpe-dlc-quant`` and ``snpe-dlc-graph-prepare``), plus inference,
benchmarking, analysis and visualization of the converted model. The
classes in this package are picked by target through the getters in
``modelconverter.packages`` and run inside the RVC4 Docker image.
"""
