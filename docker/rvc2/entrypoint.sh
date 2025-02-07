#!/bin/bash

args=("$@")
new_args=""
for arg in "${args[@]}"; do
    new_args+="\"${arg}\" "
done

set --
source $(find /opt/intel -name setupvars.sh) -pyver 3.8

if [[ "${PYTHONPATH}" != *: ]]; then
    export PYTHONPATH="${PYTHONPATH}:"
fi

if [[ -z "${new_args}" ]]; then
    exec /bin/bash
fi

eval exec modelconverter "${new_args}"
