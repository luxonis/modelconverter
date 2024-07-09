# This Dockerfile is intended for internal use only.
# If you want to build the Model Converter Docker image,
# please use the Dockerfile.public file instead.

FROM ghcr.io/luxonis/modelconverter-base:latest AS BASE
FROM python:3.10-slim

WORKDIR /app

COPY --link --from=BASE /opt/snpe /opt/snpe
COPY --link --from=BASE /usr/local/lib_310 /usr/local/lib
COPY --link --from=BASE /usr/lib /usr/lib
COPY --link --from=BASE /usr/lib_snpe /usr/lib

COPY --link docker/rvc4/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh  && \
    mkdir /app/modelconverter

COPY --link docker/rvc4/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall numpy -y && pip install 'numpy<2'

COPY --link modelconverter pyproject.toml /app/modelconverter/
RUN cd modelconverter && pip install -e . --no-deps

ENV IN_DOCKER=
ENTRYPOINT ["/app/entrypoint.sh"]
