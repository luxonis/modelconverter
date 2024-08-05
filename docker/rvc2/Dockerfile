# This Dockerfile is intended for internal use only.
# If you want to build the Model Converter Docker image,
# please use the Dockerfile.public file instead.

FROM ghcr.io/luxonis/modelconverter-base:latest AS BASE
FROM python:3.8-slim

WORKDIR /app

COPY --link --from=BASE /opt/intel /opt/intel
COPY --link --from=BASE /usr/local/bin_38 /usr/local/bin
COPY --link --from=BASE /usr/lib /usr/lib
COPY --link --from=BASE /usr/lib_openvino /usr/lib
COPY --link --from=BASE /usr/local/lib_38 /usr/local/lib

COPY --link docker/rvc2/requirements.txt .
RUN cp -r /usr/local/lib/python3.8/site-packages/openvino/libs /openvino_2023_2_libs
RUN pip install --upgrade -r requirements.txt

COPY --link docker/rvc2/entrypoint.sh /app/entrypoint.sh
COPY --link docker/rvc2/compile_tool /usr/local/bin/compile_tool

RUN chmod +x /usr/local/bin/compile_tool && \
    chmod +x /app/entrypoint.sh && \
    mkdir /app/modelconverter

COPY --link modelconverter pyproject.toml /app/modelconverter/
RUN cd modelconverter && pip install -e . --no-deps

ENV IN_DOCKER=
ENTRYPOINT ["/app/entrypoint.sh"]
