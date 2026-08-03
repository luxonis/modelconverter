#!/bin/bash

if [[ $PYTHONPATH != *: ]]; then
    export PYTHONPATH=$PYTHONPATH:
fi

source "/opt/snpe/bin/envsetup.sh"

source "$(dirname "${BASH_SOURCE[0]}")/entrypoint_common.sh"
