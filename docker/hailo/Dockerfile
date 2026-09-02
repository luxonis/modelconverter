ARG VERSION=2025.04
ARG DALI_VERSION=1.49.0

FROM hailo_ai_sw_suite_${VERSION//./-}:1

USER root

ARG VERSION
ARG DALI_VERSION

ENV VERSION=${VERSION//./-}
ENV IN_DOCKER=

WORKDIR /app

COPY --link --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
COPY --link pyproject.toml uv.lock ./
COPY --link modelconverter/platforms/hailo/requirements.txt requirements-hailo.txt

RUN <<EOF

    set -e

    # pip installs the DALI archives below; uv installs everything else.
    # pip 26 ignores the non-PEP517 install path we need for the DALI TF
    # plugin archive and falls back to an isolated build that crashes on
    # missing pkg_resources.
    pip install --no-cache-dir "pip==25.0.1"

    # The suite image ships its own Python environment. Aim uv at the same
    # interpreter the suite's `pip` and `python` use.
    PYTHON="$(command -v python)"

    # Install the shared dependencies exactly as the lockfile pins them.
    uv export --frozen --no-default-groups --no-emit-project \
        -o /tmp/requirements.txt
    uv pip install --python "$PYTHON" --no-cache --no-deps \
        -r /tmp/requirements.txt

    # The suite's TensorFlow 2.12 is compiled against the NumPy 1.x ABI and
    # fails to import under NumPy 2, which luxonis-ml requires from 0.9.1 on.
    # The locked install above brings NumPy 2 in. Step NumPy back down here,
    # before anything imports TensorFlow. Extensions built against NumPy 2
    # keep working on 1.x; only the reverse breaks. The downgrade
    # deliberately violates luxonis-ml's pin. uv resolves only what each
    # command asks for, so the rest of the environment stays as it is.
    uv pip install --python "$PYTHON" --no-cache "numpy<2"
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
    uv pip install --python "$PYTHON" --no-cache -r requirements-hailo.txt

EOF

ARG DEV=false
RUN if [ "$DEV" = "true" ]; then \
        uv pip install --python "$(command -v python)" --no-cache \
            --group dev codecov-cli; \
    fi

COPY --link modelconverter /app/modelconverter/
RUN uv pip install --python "$(command -v python)" --no-cache --no-deps -e .

COPY --link docker/hailo/entrypoint.sh /app/entrypoint.sh
COPY --link docker/entrypoint_common.sh /app/entrypoint_common.sh

RUN <<EOF

    set -e

    echo "export SSL_CERT_FILE=$(python -m certifi)" > /etc/profile.d/certifi.sh
    chmod +x /app/entrypoint.sh

EOF

ENTRYPOINT ["/app/entrypoint.sh"]
