ARG VERSION=2025.04
ARG DALI_VERSION=1.49.0

FROM hailo_ai_sw_suite_${VERSION//./-}:1

USER root

ARG VERSION
ARG DALI_VERSION

ENV VERSION=${VERSION//./-}
ENV IN_DOCKER=

WORKDIR /app

COPY --link requirements.txt requirements.txt
COPY --link modelconverter/packages/hailo/requirements.txt requirements-hailo.txt

RUN <<EOF

    set -e

    pip install --upgrade pip --no-cache-dir
    pip install -r requirements.txt --no-cache-dir
    python - <<PY
from urllib.request import urlretrieve

version = "${DALI_VERSION}"
base_url = "https://developer.download.nvidia.com/compute/redist"
artifacts = [
    (
        f"{base_url}/nvidia-dali-cuda120/"
        f"nvidia_dali_cuda120-{version}-py3-none-manylinux2014_x86_64.whl",
        f"/tmp/nvidia_dali_cuda120-{version}-py3-none-manylinux2014_x86_64.whl",
    ),
    (
        f"{base_url}/nvidia-dali-tf-plugin-cuda120/"
        f"nvidia_dali_tf_plugin_cuda120-{version}.tar.gz",
        f"/tmp/nvidia_dali_tf_plugin_cuda120-{version}.tar.gz",
    ),
]

for url, dest in artifacts:
    urlretrieve(url, dest)
PY
    pip install --no-cache-dir --no-deps \
        "/tmp/nvidia_dali_cuda120-${DALI_VERSION}-py3-none-manylinux2014_x86_64.whl"
    PIP_USE_PEP517=0 pip install --no-cache-dir --no-deps \
        "/tmp/nvidia_dali_tf_plugin_cuda120-${DALI_VERSION}.tar.gz"
    pip install --no-cache-dir -r requirements-hailo.txt

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
