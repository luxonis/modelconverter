FROM hailo_ai_sw_suite_2024-07:1
USER root

RUN rm -rf \
    /local/workspace/tappas \
    /opt/google \
    /local/workspace/doc \
    /local/workspace/hailort_examples \
    /usr/share \
    /usr/bin/docker* \
    /usr/bin/containerd* \
    /local/workspace/hailo_virtualenv/lib/python3.8/site-packages/hailo_tutorials

COPY requirements.txt .

RUN <<EOF

    set -e

    pip install --upgrade pip
    pip install -r requirements.txt
    pip install --extra-index-url
        https://developer.download.nvidia.com/compute/redist \
        nvidia-dali-cuda110
    pip install --extra-index-url \
        https://developer.download.nvidia.com/compute/redist \
        nvidia-dali-tf-plugin-cuda110

EOF

RUN <<EOF

    set -e

    pip install pip-autoremove

    pip-autoremove -y \
        torch         \
        jupyter       \
        plotly        \
        matplotlib    \
        ipython       \
        tensorboard   \
        pip-autoremove

    pip install psutil
    rm -rf ~/.cache

EOF

WORKDIR /app

COPY docker/hailo/entrypoint.sh /app/entrypoint.sh
COPY --link modelconverter /app/modelconverter/
COPY --link pyproject.toml /app/pyproject.toml

RUN <<EOF

    set -e

    pip install -e . --no-deps --no-cache-dir
    echo "export SSL_CERT_FILE=$(python -m certifi)" > /etc/profile.d/certifi.sh
    chmod +x /app/entrypoint.sh

EOF


ENTRYPOINT ["/app/entrypoint.sh"]
