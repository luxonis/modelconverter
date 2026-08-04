# Shared part of every target entrypoint: run the requested command as a
# signal-forwarding child, then hand the mounts back to the invoking host user.
#
# Sourced -- not executed -- by docker/<target>/entrypoint.sh after that
# target's own environment setup, with the container's arguments intact. The
# Dockerfiles copy it next to the entrypoint, at /app/entrypoint_common.sh.

# The conversion tools run as root, so anything written to the cache or the
# output directory would otherwise be owned by root on the host.
chown_mounts() {
    read -r container_uid host_uid _ < /proc/self/uid_map
    if [[ -z "$HOST_UID" || -z "$HOST_GID" ||
          "$container_uid" != 0 || "$host_uid" != 0 ]]; then
        return
    fi
    # `/app/tests` is bind-mounted into dev containers, where running the suite
    # leaves `__pycache__` and report files behind in the host checkout.
    chown -R "$HOST_UID:$HOST_GID" /app/output /app/tests 2>/dev/null || true
    # The staged inputs are written by the host user and only read in here, and
    # they are the part of the cache that grows: walking them after every run
    # would cost more the more inputs have been cached.
    find /app/shared_with_container -mindepth 1 -maxdepth 1 ! -name inputs \
        -exec chown -R "$HOST_UID:$HOST_GID" {} + 2>/dev/null || true
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
