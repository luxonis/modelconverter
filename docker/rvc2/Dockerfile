FROM python:3.8-slim as BASE

ARG VERSION=2022.3.0
ENV VERSION=${VERSION}


RUN <<EOF

apt-get update
apt-get install -y cmake unzip perl libatomic1 \
    libc++-dev ffmpeg libcurl4 libncurses5 llvm-14-runtime patch git

EOF

COPY --link requirements.txt .
COPY docker/rvc2/requirements.txt requirements-rvc2.txt
COPY --link docker/extra_packages/openvino-${VERSION}.tar.gz .
COPY --link docker/extra_packages/mo_patch.diff .

RUN <<EOF

pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -r requirements-rvc2.txt
if [ "${VERSION}" = "2021.4.0" ]; then
    pip install --no-cache-dir onnxruntime 'numpy<1.20'
fi
mkdir /opt/intel

if [ "${VERSION}" = "2022.3.0" ]; then
    tar xvf "openvino-${VERSION}.tar.gz" -C /opt/intel --strip-components 1
    patch /usr/local/lib/python3.8/site-packages/openvino/tools/mo/convert_impl.py < /mo_patch.diff
else
    tar xvf "openvino-${VERSION}.tar.gz" -C /opt/intel
fi

EOF


# RUN sed -i 's/libtbb2/libtbbmalloc2/g' \
#     /opt/intel/install_dependencies/install_openvino_dependencies.sh && \
#     bash /opt/intel/install_dependencies/install_openvino_dependencies.sh -y


FROM python:3.8-slim

ENV IN_DOCKER=

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

COPY --link docker/rvc2/entrypoint.sh /app/entrypoint.sh

RUN <<EOF

COMPILE_TOOL=$(find /opt/intel -name compile_tool -type f)
echo "alias compile_tool=${COMPILE_TOOL}"  >> ~/.bashrc
chmod +x /app/entrypoint.sh
mkdir /app/modelconverter
EOF

COPY --link modelconverter pyproject.toml /app/modelconverter/
RUN cd modelconverter && pip install -e . --no-deps --no-cache-dir

ENTRYPOINT ["/app/entrypoint.sh"]
