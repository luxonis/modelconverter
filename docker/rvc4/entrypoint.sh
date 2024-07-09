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

if [[ -z $new_args ]]; then
    exec /bin/bash
fi

eval exec modelconverter $new_args
