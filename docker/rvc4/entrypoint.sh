#!/bin/bash

if [[ $PYTHONPATH != *: ]]; then
    export PYTHONPATH=$PYTHONPATH:
fi

source "/opt/snpe/bin/envsetup.sh"

# Chown the writable mounts back to the invoking host user. The conversion
# tools run as root, so anything written to the cache or the output directory
# would otherwise be owned by root on the host.
chown_mounts() {
    read -r container_uid host_uid _ < /proc/self/uid_map
    if [[ -n "$HOST_UID" && -n "$HOST_GID" &&
          "$container_uid" == 0 && "$host_uid" == 0 ]]; then
        chown -R "$HOST_UID:$HOST_GID" \
            /app/shared_with_container /app/output 2>/dev/null || true
    fi
}

child_pid=""
forward_signal() {
    if [[ -n "$child_pid" ]]; then
        kill "-$1" "$child_pid" 2>/dev/null || true
    fi
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

if [[ $# -eq 0 ]]; then
    /bin/bash <&0 &
else
    modelconverter "$@" &
fi
child_pid=$!

while true; do
    wait "$child_pid"
    status=$?
    if ! kill -0 "$child_pid" 2>/dev/null; then
        break
    fi
done

chown_mounts
exit $status
