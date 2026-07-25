#!/bin/bash

args=("$@")
new_args=""
for arg in "${args[@]}"; do
    new_args+="\"$arg\" "
done

set --

if [[ $PYTHONPATH != *: ]]; then
    export PYTHONPATH=$PYTHONPATH:
fi

source "/opt/snpe/bin/envsetup.sh"

# Chown the writable mounts back to the invoking host user. The conversion
# tools run as root, so anything written to the cache or the output directory
# would otherwise be owned by root on the host.
chown_mounts() {
    if [[ -n "$HOST_UID" && -n "$HOST_GID" ]]; then
        chown -R "$HOST_UID:$HOST_GID" \
            /app/shared_with_container /app/output 2>/dev/null || true
    fi
}

if [[ -z $new_args ]]; then
    /bin/bash
    status=$?
    chown_mounts
    exit $status
fi

eval modelconverter $new_args
status=$?
chown_mounts
exit $status
