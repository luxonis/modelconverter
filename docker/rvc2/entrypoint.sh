#!/bin/bash

args=("$@")
new_args=""
for arg in "${args[@]}"; do
    new_args+="\"${arg}\" "
done

set --


if [ ${VERSION} = "2021.4.0" ]; then
    source /opt/intel/bin/setupvars.sh -pyver 3.8
else
    source /opt/intel/setupvars.sh -pyver 3.8
fi

if [[ "${PYTHONPATH}" != *: ]]; then
    export PYTHONPATH="${PYTHONPATH}:"
fi

if [[ -z "${new_args}" ]]; then
    exec /bin/bash
fi

eval exec modelconverter "${new_args}"
