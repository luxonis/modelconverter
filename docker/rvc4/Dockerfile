FROM python:3.10-slim AS base

ARG VERSION=2.41.0
ENV VERSION=${VERSION}

RUN <<EOF

    set -e

    echo 'deb http://deb.debian.org/debian oldstable main' >> /etc/apt/sources.list

    apt-get update -qq
    apt-get install -y -qq --fix-missing \
        cmake           \
        unzip           \
        perl            \
        libatomic1      \
        libc++-dev      \
        ffmpeg          \
        libcurl4t64     \
        libncurses5     \
        libtinfo5       \
        llvm-14-runtime

EOF

COPY --link docker/extra_packages/snpe-${VERSION}.zip snpe.zip

RUN <<EOF

    set -e

    unzip -q snpe.zip -d /opt

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

COPY --link --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
COPY --link pyproject.toml uv.lock ./
COPY --link modelconverter/platforms/rvc4/requirements.txt requirements-rvc4.txt

# Install the shared dependencies exactly as the lockfile pins them. The
# platform pins (including the NumPy downgrade for SNPE) then resolve on
# top.
RUN uv export --frozen --no-default-groups --no-emit-project \
    -o /tmp/requirements.txt
RUN uv pip install --system --no-cache --no-deps -r /tmp/requirements.txt
RUN uv pip install --system --no-cache -r requirements-rvc4.txt

FROM python:3.10-slim

ARG VERSION
ENV IN_DOCKER=
ENV VERSION=${VERSION}

WORKDIR /app

COPY --link --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

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
    /usr/lib/x86_64-linux-gnu/libtinfo.so.5 \
    /usr/lib/x86_64-linux-gnu/libLLVM-14.so.1 \
    /usr/lib/


COPY --link docker/rvc4/entrypoint.sh /app/entrypoint.sh
COPY --link docker/entrypoint_common.sh /app/entrypoint_common.sh

ARG DEV=false
COPY --link requirements-eval.txt /app/requirements-eval.txt
COPY --link pyproject.toml /app/pyproject.toml
RUN if [ "$DEV" = "true" ]; then \
        apt-get update -qq && \
        apt-get install -y -qq --no-install-recommends git && \
        rm -rf /var/lib/apt/lists/* && \
        uv pip install --system --no-cache --group dev codecov-cli && \
        uv pip install --system --no-cache --no-deps -r /app/requirements-eval.txt; \
    fi

COPY --link modelconverter /app/modelconverter/
RUN <<EOF

    set -e

    uv pip install --system --no-cache --no-deps -e .
    chmod +x /app/entrypoint.sh

EOF

ENTRYPOINT ["/app/entrypoint.sh"]
