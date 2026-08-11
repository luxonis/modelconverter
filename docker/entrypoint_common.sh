# Shared part of every target entrypoint: run the requested command as a
# signal-forwarding child, then hand the mounts back to the invoking host user.
#
# Sourced -- not executed -- by docker/<target>/entrypoint.sh after that
# target's own environment setup, with the container's arguments intact. The
# Dockerfiles copy it next to the entrypoint, at /app/entrypoint_common.sh.

# Where the image puts the mounts. Only ever overridden by the tests, which
# cannot let a `chown -R` loose on a real `/app`.
APP_ROOT="${APP_ROOT:-/app}"

# The conversion tools run as root, so anything written to the cache or the
# output directory would otherwise be owned by root on the host.
chown_mounts() {
    read -r container_uid host_uid _ < /proc/self/uid_map
    if [[ -z "$HOST_UID" || -z "$HOST_GID" ||
          "$container_uid" != 0 || "$host_uid" != 0 ]]; then
        return
    fi
    owner="$HOST_UID:$HOST_GID"
    # `tests` and `modelconverter` are bind-mounted into dev containers, where
    # running the suite leaves `__pycache__` and report files behind in the
    # host checkout. Neither exists in a plain conversion.
    for mount in "$APP_ROOT/output" "$APP_ROOT/tests" \
                 "$APP_ROOT/modelconverter"; do
        [[ -d $mount ]] && chown -R "$owner" "$mount" 2>/dev/null
    done
    # The cache mount root too: when its host directory did not exist, the
    # daemon created it as root and nothing else ever repairs that -- the next
    # run would fail to stage its inputs into it.
    chown "$owner" "$APP_ROOT/shared_with_container" 2>/dev/null
    # The staged inputs are written by the host user and only read in here, and
    # they are the part of the cache that grows: walking them after every run
    # would cost more the more inputs have been cached.
    find "$APP_ROOT/shared_with_container" -mindepth 1 -maxdepth 1 \
        ! -name inputs -exec chown -R "$owner" {} + 2>/dev/null
    return 0
}

child_pid=""
signal_interrupted=""
forward_signal() {
    signal_interrupted=1
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

# A trapped signal interrupts `wait` before the child has necessarily exited --
# possibly in the very instant the child exits, in which case `wait` reports
# 128+signal even though the child's own status was something else. Only a
# round no trap interrupted is trusted; after an interrupted one, waiting
# again either keeps waiting for the live child or has bash report the exit
# status it remembers for a reaped one.
while true; do
    signal_interrupted=""
    wait "$child_pid"
    status=$?
    if [[ -z "$signal_interrupted" ]]; then
        break
    fi
done

chown_mounts
exit $status
