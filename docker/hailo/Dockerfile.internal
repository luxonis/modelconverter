# This Dockerfile is intended for internal use only.
# If you want to build the Model Converter Docker image,
# please use the Dockerfile.public file instead.

ARG VERSION=2.23.0

FROM ghcr.io/luxonis/modelconverter-hailo:${VERSION}-base

ENV IN_DOCKER=
ENV VERSION=${VERSION}

COPY requirements.txt requirements.txt

RUN <<EOF

    set -e

    pip install --upgrade pip
    pip install -r requirements.txt

EOF

WORKDIR /app

COPY --link docker/hailo/entrypoint.sh /app/entrypoint.sh
COPY --link modelconverter /app/modelconverter/
COPY --link pyproject.toml /app/pyproject.toml

RUN <<EOF

    set -e

    echo "export SSL_CERT_FILE=$(python -m certifi)" \
        > /etc/profile.d/certifi.sh

    pip install -e . --no-deps --no-cache-dir
    chmod +x /app/entrypoint.sh

EOF

ENTRYPOINT ["/app/entrypoint.sh"]
