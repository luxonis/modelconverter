ARG VERSION=2025.04

FROM hailo_ai_sw_suite_${VERSION//./-}:1

USER root

ARG VERSION

ENV VERSION=${VERSION//./-}
ENV IN_DOCKER=

WORKDIR /app

COPY --link requirements.txt requirements.txt
COPY --link modelconverter/packages/hailo/requirements.txt requirements-hailo.txt

RUN <<EOF

    set -e

    pip install --upgrade pip --no-cache-dir
    pip install -r requirements.txt --no-cache-dir
    pip install --extra-index-url \
        https://developer.download.nvidia.com/compute/redist \
        nvidia-dali-cuda120 --no-cache-dir
    pip install --extra-index-url \
        https://developer.download.nvidia.com/compute/redist \
        nvidia-dali-tf-plugin-cuda120 --no-cache-dir
    pip uninstall protobuf -y
    pip install protobuf==3.20.3 --no-cache-dir

EOF

COPY --link modelconverter /app/modelconverter/
COPY --link pyproject.toml /app/pyproject.toml

RUN pip install -e . --no-deps --no-cache-dir

COPY --link docker/hailo/entrypoint.sh /app/entrypoint.sh

RUN <<EOF

    set -e

    echo "export SSL_CERT_FILE=$(python -m certifi)" > /etc/profile.d/certifi.sh
    chmod +x /app/entrypoint.sh

EOF

ENTRYPOINT ["/app/entrypoint.sh"]
