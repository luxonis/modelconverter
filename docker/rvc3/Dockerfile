FROM python:3.8-slim AS base

RUN <<EOF

    set -e

    apt-get update
    apt-get install -y  \
        cmake           \
        unzip           \
        perl            \
        libatomic1      \
        libc++-dev      \
        ffmpeg libcurl4 \
        libncurses5     \
        llvm-14-runtime \
        patch           \
        git

EOF

COPY --link docker/extra_packages/openvino-2022.3.0.tar.gz .

COPY --link requirements.txt requirements.txt
COPY --link docker/rvc3/requirements.txt requirements-rvc3.txt

RUN <<EOF

    set -e

    mkdir /opt/intel
    tar xvf openvino-2022.3.0.tar.gz -C /opt/intel/ --strip-components 1

EOF

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install /opt/intel/tools/*.whl
RUN pip install -r requirements-rvc3.txt

COPY --link docker/patches /patches

RUN <<EOF

    set -e

    # Model Optimizer contains a bug in both versions
    # that needs to be patched
    if [ "${VERSION}" = "2022.3.0" ]; then
        patch /usr/local/lib/python3.8/site-packages/openvino/tools/mo/convert_impl.py \
            < /patches/mo-2022.3.patch
    else
        patch /opt/intel/deployment_tools/model_optimizer/mo/main.py \
            < /patches/mo-2021.4.patch
    fi
    rm -r /opt/intel/tools/*.whl

EOF


FROM python:3.8-slim

ENV IN_DOCKER=

WORKDIR /app

COPY --link --from=base \
    /usr/lib/x86_64-linux-gnu/libcurl.so.4 \
    /usr/lib/x86_64-linux-gnu/libnghttp2.so.14 \
    /usr/lib/x86_64-linux-gnu/librtmp.so.1 \
    /usr/lib/x86_64-linux-gnu/libssh2.so.1 \
    /usr/lib/x86_64-linux-gnu/libpsl.so.5 \
    /usr/lib/x86_64-linux-gnu/libldap-2.5.so.0 \
    /usr/lib/x86_64-linux-gnu/liblber-2.5.so.0 \
    /usr/lib/x86_64-linux-gnu/libbrotlidec.so.1 \
    /usr/lib/x86_64-linux-gnu/libsasl2.so.2 \
    /usr/lib/x86_64-linux-gnu/libbrotlicommon.so.1 \
    /usr/lib/x86_64-linux-gnu/libGL.so.1 \
    /usr/lib/x86_64-linux-gnu/libgthread-2.0.so.0 \
    /usr/lib/x86_64-linux-gnu/libglib-2.0.so.0 \
    /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 \
    /usr/lib/x86_64-linux-gnu/libGLX.so.0 \
    /usr/lib/x86_64-linux-gnu/libX11.so.6 \
    /usr/lib/x86_64-linux-gnu/libxcb.so.1 \
    /usr/lib/x86_64-linux-gnu/libXau.so.6 \
    /usr/lib/x86_64-linux-gnu/libXdmcp.so.6 \
    /usr/lib/x86_64-linux-gnu/libbsd.so.0 \
    /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 \
    /usr/lib/

COPY --link --from=base /opt/intel /opt/intel
COPY --link --from=base /usr/local/bin /usr/local/bin
COPY --link --from=base /usr/local/lib /usr/local/lib

RUN pip install --upgrade pip

COPY --link docker/rvc3/entrypoint.sh /app/entrypoint.sh
COPY --link modelconverter /app/modelconverter/
COPY --link pyproject.toml /app/pyproject.toml

RUN <<EOF

    set -e

    COMPILE_TOOL=/opt/intel/tools/compile_tool/compile_tool

    pip install -e . --no-deps --no-cache-dir

    echo "alias compile_tool=${COMPILE_TOOL}" >> ~/.bashrc
    chmod +x /app/entrypoint.sh

EOF


ENTRYPOINT ["/app/entrypoint.sh"]
