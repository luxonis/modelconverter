#!/bin/bash

args=("$@")
new_args=""
for arg in "${args[@]}"; do
    new_args+="\"$arg\" "
done

if [[ "${args[0]}" != "infer" ]]; then
    set --
    source $(find /opt/intel -name setupvars.sh) -pyver 3.8
fi

if [[ "${PYTHONPATH}" != *: ]]; then
    export PYTHONPATH="${PYTHONPATH}:"
fi

if [[ -z $new_args ]]; then
    exec /bin/bash
fi

eval exec modelconverter $new_args
