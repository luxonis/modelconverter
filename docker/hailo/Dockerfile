ARG VERSION=2024.04

FROM hailo_ai_sw_suite_${VERSION//./-}:1

USER root

ARG VERSION

ENV VERSION=${VERSION//./-}
ENV IN_DOCKER=

WORKDIR /app

RUN rm -rf \
    /local/workspace/tappas \
    /opt/google \
    /local/workspace/doc \
    /local/workspace/hailort_examples \
    /usr/share \
    /usr/bin/docker* \
    /usr/bin/containerd* \
    /local/workspace/hailo_virtualenv/lib/python3.8/site-packages/hailo_tutorials

COPY --link requirements.txt requirements.txt
COPY --link modelconverter/packages/hailo/requirements.txt requirements-hailo.txt

RUN <<EOF

    set -e

    pip install --upgrade pip --no-cache-dir
    pip install -r requirements.txt --no-cache-dir
    pip install --extra-index-url \
        https://developer.download.nvidia.com/compute/redist \
        nvidia-dali-cuda110 --no-cache-dir
    pip install --extra-index-url \
        https://developer.download.nvidia.com/compute/redist \
        nvidia-dali-tf-plugin-cuda110 --no-cache-dir

EOF

RUN <<EOF

    set -e

    if [[ "${VERSION}" = "2024-04" ]]; then
        pip install --upgrade numpy==1.22.0 --no-cache-dir
    fi

    pip install psutil  --no-cache-dir

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
