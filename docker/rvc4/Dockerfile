FROM python:3.10-slim AS base

ARG VERSION=2.23.0
ENV VERSION=${VERSION}

RUN <<EOF

    set -e

    apt-get update
    apt-get install -y --fix-missing \
        cmake           \
        unzip           \
        perl            \
        libatomic1      \
        libc++-dev      \
        ffmpeg          \
        libcurl4        \
        libncurses6     \
        llvm-19-runtime \
        git

EOF

COPY --link docker/extra_packages/snpe-${VERSION}.zip snpe.zip

RUN <<EOF

    set -e

    unzip snpe.zip -d /opt

    # The SNPE archive from README.md contains an extra directory
    if [ -d /opt/qairt ]; then
        mv /opt/qairt/* /opt/
        rmdir /opt/qairt
    fi

    mv /opt/* /opt/snpe

    rm -rf /opt/snpe/Uninstall
    rm -rf /opt/snpe/docs
    rm -rf /opt/snpe/share
    rm -rf /opt/snpe/examples
    rm -rf /opt/snpe/benchmarks
    rm -rf /opt/snpe/lib/hexagon*
    rm -rf /opt/snpe/lib/*android*
    rm -rf /opt/snpe/lib/*ubuntu*
    rm -rf /opt/snpe/lib/*windows*
    rm -rf /opt/snpe/bin/*windows*
    rm -rf /opt/snpe/bin/*android*
    rm -rf /opt/snpe/bin/*ubuntu*

EOF

COPY --link requirements.txt requirements.txt
COPY --link modelconverter/packages/rvc4/requirements.txt requirements-rvc4.txt

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install --upgrade -r requirements-rvc4.txt

FROM python:3.10-slim

ARG VERSION
ENV IN_DOCKER=
ENV VERSION=${VERSION}

WORKDIR /app

RUN pip install --upgrade pip

COPY --link --from=base /opt/ /opt/
COPY --link --from=base /usr/local/lib /usr/local/lib
COPY --link --from=base \
    /usr/lib/x86_64-linux-gnu/libcurl.so.4 \
    /usr/lib/x86_64-linux-gnu/libnghttp2.so.14 \
    /usr/lib/x86_64-linux-gnu/librtmp.so.1 \
    /usr/lib/x86_64-linux-gnu/libssh2.so.1 \
    /usr/lib/x86_64-linux-gnu/libpsl.so.5 \
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
    /usr/lib/x86_64-linux-gnu/libc++.so.1 \
    /usr/lib/x86_64-linux-gnu/libc++abi.so.1 \
    /usr/lib/x86_64-linux-gnu/libunwind.so.1 \
    /usr/lib/x86_64-linux-gnu/libatomic.so.1 \
    /usr/lib/x86_64-linux-gnu/libedit.so.2 \
    /usr/lib/x86_64-linux-gnu/libz3.so.4 \
    /usr/lib/x86_64-linux-gnu/libxml2.so.2 \
    /usr/lib/


COPY --link docker/rvc4/entrypoint.sh /app/entrypoint.sh
COPY --link modelconverter /app/modelconverter/
COPY --link pyproject.toml /app/pyproject.toml

RUN <<EOF

    set -e

    pip install -e . --no-deps --no-cache-dir
    chmod +x /app/entrypoint.sh

EOF

ENTRYPOINT ["/app/entrypoint.sh"]
