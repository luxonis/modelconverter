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

    # pip 26 ignores the non-PEP517 install path we need for the DALI TF
    # plugin archive and falls back to an isolated build that crashes on
    # missing pkg_resources.
    pip install --no-cache-dir "pip==25.0.1"
    pip install -r requirements.txt --no-cache-dir

    # The suite's TensorFlow 2.12 is compiled against the NumPy 1.x ABI and
    # fails to import under NumPy 2, which luxonis-ml requires from 0.9.1 on.
    # Resolve the requirements against NumPy 2 so luxonis-ml stays current, then
    # step NumPy back down before anything imports TensorFlow: extensions built
    # against NumPy 2 keep working on 1.x, only the reverse breaks. pip reports
    # the deliberate mismatch with luxonis-ml's pin as a warning and carries on.
    pip install --no-cache-dir "numpy<2"
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

ARG DEV=false
COPY --link requirements-dev.txt /app/requirements-dev.txt
RUN if [ "$DEV" = "true" ]; then \
        pip install --no-cache-dir -r /app/requirements-dev.txt codecov-cli; \
    fi

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
