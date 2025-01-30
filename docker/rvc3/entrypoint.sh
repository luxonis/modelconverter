#!/bin/bash

args=("$@")
new_args=""
for arg in "${args[@]}"; do
    new_args+="\"$arg\" "
done

set --
source /opt/intel/setupvars.sh -pyver 3.8


if [[ -z $new_args ]]; then
    exec /bin/bash
fi

eval exec modelconverter $new_args
