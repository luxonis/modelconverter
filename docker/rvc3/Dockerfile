FROM python:3.8-slim as BASE

RUN apt-get update && \
    apt-get install -y \
    cmake unzip perl libatomic1 libc++-dev ffmpeg libcurl4 libncurses5 llvm-14-runtime patch git

COPY --link docker/scripts /scripts
RUN bash /scripts/install_openssl.sh

COPY --link docker/extra_packages/openvino_2022_3_vpux_drop_patched.tar.gz .
RUN mkdir /opt/intel
RUN tar xvf openvino_2022_3_vpux_drop_patched.tar.gz \
    -C /opt/intel/ --strip-components 1
RUN sed -i 's/libtbb2/libtbbmalloc2/g' \
    /opt/intel/install_dependencies/install_openvino_dependencies.sh && \
    bash /opt/intel/install_dependencies/install_openvino_dependencies.sh -y

RUN pip install --upgrade pip
COPY --link requirements.txt .
RUN pip install -r requirements.txt
RUN pip install /opt/intel/tools/*.whl
COPY --link docker/extra_packages/mo_patch.diff .
RUN patch /usr/local/lib/python3.8/site-packages/openvino/tools/mo/convert_impl.py < /mo_patch.diff

COPY docker/rvc3/requirements.txt .

RUN pip install -r requirements.txt

FROM python:3.8-slim

WORKDIR /app

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
    /usr/lib/x86_64-linux-gnu/libusb-1.0.so.0 \
    /usr/lib/

COPY --link --from=BASE /opt/intel /opt/intel
COPY --link --from=BASE /usr/local/bin /usr/local/bin
COPY --link --from=BASE /usr/local/lib /usr/local/lib


COPY --link docker/rvc3/entrypoint.sh /app/entrypoint.sh

RUN echo "alias compile_tool='/opt/intel/tools/compile_tool/compile_tool'"  \
    >> ~/.bashrc && \
    chmod +x /app/entrypoint.sh && \
    mkdir /app/modelconverter

COPY --link modelconverter pyproject.toml /app/modelconverter/
RUN cd modelconverter && pip install -e . --no-deps --no-cache-dir

ENV IN_DOCKER=
ENTRYPOINT ["/app/entrypoint.sh"]
