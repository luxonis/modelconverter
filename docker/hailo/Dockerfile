# This Dockerfile is intended for internal use only.
# If you want to build the Model Converter Docker image,
# please use the Dockerfile.public file instead.

FROM ghcr.io/luxonis/modelconverter-base-hailo:latest

WORKDIR /app
COPY docker/hailo/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

COPY modelconverter pyproject.toml requirements.txt /app/modelconverter/
RUN cd modelconverter
RUN cd modelconverter && pip install -e . --no-deps

RUN echo "export SSL_CERT_FILE=$(python -m certifi)" > /etc/profile.d/certifi.sh

ENV IN_DOCKER=
ENTRYPOINT ["/app/entrypoint.sh"]
