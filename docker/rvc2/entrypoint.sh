#!/bin/bash

args=("$@")
new_args=""
for arg in "${args[@]}"; do
    new_args+="\"$arg\" "
done

set --
if [ -f /opt/intel/setupvars.sh ]; then
    source /opt/intel/setupvars.sh
elif [ -f /opt/intel/*openvino*/bin/setupvars.sh ]; then
    source /opt/intel/*openvino*/bin/setupvars.sh
fi

if [[ $PYTHONPATH != *: ]]; then
    export PYTHONPATH=$PYTHONPATH:
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/site-packages/openvino/libs/

if [[ -z $new_args ]]; then
    exec /bin/bash
fi

eval exec modelconverter $new_args
