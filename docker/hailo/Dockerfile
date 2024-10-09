# This Dockerfile is intended for internal use only.
# If you want to build the Model Converter Docker image,
# please use the Dockerfile.public file instead.

ARG VERSION=2.23.0

FROM ghcr.io/luxonis/modelconverter-hailo:${VERSION}-base

ENV IN_DOCKER=
ENV VERSION=${VERSION}

WORKDIR /app

COPY --link docker/hailo/entrypoint.sh /app/entrypoint.sh
COPY --link modelconverter pyproject.toml /app/modelconverter/

RUN <<EOF

    echo "export SSL_CERT_FILE=$(python -m certifi)" \
        > /etc/profile.d/certifi.sh

    pip install -e ./modelconverter --no-deps --no-cache-dir
    chmod +x /app/entrypoint.sh

EOF

ENTRYPOINT ["/app/entrypoint.sh"]
