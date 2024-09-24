FROM python:3.10-slim as BASE

RUN apt-get update && \
    apt-get install -y \
    cmake unzip perl libatomic1 libc++-dev ffmpeg \
    libcurl4 libncurses5 llvm-14-runtime git

COPY --link docker/extra_packages/snpe.zip .
RUN unzip snpe.zip -d /opt

RUN rm -rf /opt/snpe/Uninstall && \
    rm -rf /opt/snpe/docs && \
    rm -rf /opt/snpe/examples && \
    rm -rf /opt/snpe/lib/hexagon* && \
    rm -rf /opt/snpe/lib/*android* && \
    rm -rf /opt/snpe/lib/*ubuntu* && \
    rm -rf /opt/snpe/lib/*windows* && \
    rm -rf /opt/snpe/bin/*windows* && \
    rm -rf /opt/snpe/bin/*android* && \
    rm -rf /opt/snpe/bin/*ubuntu*

RUN pip install --upgrade pip
COPY --link requirements.txt .
RUN pip install -r requirements.txt

COPY --link docker/rvc4/requirements.txt .
RUN pip install --upgrade -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --link --from=BASE /opt/snpe /opt/snpe
COPY --link --from=BASE /usr/local/lib /usr/local/lib
COPY --link --from=BASE \
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
    /usr/lib/x86_64-linux-gnu/libc++.so.1 \
    /usr/lib/x86_64-linux-gnu/libc++abi.so.1 \
    /usr/lib/x86_64-linux-gnu/libunwind.so.1 \
    /usr/lib/x86_64-linux-gnu/libatomic.so.1 \
    /usr/lib/x86_64-linux-gnu/libtinfo.so.5 \
    /usr/lib/x86_64-linux-gnu/libLLVM-14.so.1 \
    /usr/lib/x86_64-linux-gnu/libedit.so.2 \
    /usr/lib/x86_64-linux-gnu/libz3.so.4 \
    /usr/lib/x86_64-linux-gnu/libxml2.so.2 \
    /usr/lib/x86_64-linux-gnu/libicuuc.so.72 \
    /usr/lib/x86_64-linux-gnu/libicudata.so.72 \
    /usr/lib/

COPY docker/rvc4/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh  && \
    mkdir /app/modelconverter

COPY --link modelconverter pyproject.toml /app/modelconverter/
RUN cd modelconverter && pip install -e . --no-deps

ENV IN_DOCKER=
ENTRYPOINT ["/app/entrypoint.sh"]
