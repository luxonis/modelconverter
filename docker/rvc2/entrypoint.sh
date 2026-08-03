#!/bin/bash

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

# The child runs asynchronously so the traps above stay responsive while it
# works. Bash then applies two defaults we have to undo: an asynchronous
# command gets its stdin redirected from /dev/null (hence `<&0`), and, with job
# control off, its SIGINT/SIGQUIT set to SIG_IGN. SIG_IGN survives execve and
# CPython only installs its own handler when it inherits SIG_DFL, so without the
# `trap -` reset a forwarded Ctrl-C would be silently ignored by the converter.
if [[ $# -eq 0 ]]; then
    ( trap - INT QUIT; exec /bin/bash ) <&0 &
else
    ( trap - INT QUIT; exec modelconverter "$@" ) <&0 &
fi
child_pid=$!

# A trapped signal interrupts `wait` before the child has necessarily exited.
# Keep waiting so the converter can finish its own signal cleanup.
while true; do
    wait "$child_pid"
    status=$?
    if ! kill -0 "$child_pid" 2>/dev/null; then
        break
    fi
done

chown_mounts
exit $status
