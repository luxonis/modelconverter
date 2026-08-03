#!/bin/bash

source /etc/profile.d/certifi.sh

if [[ $PYTHONPATH != *: ]]; then
    export PYTHONPATH=$PYTHONPATH:
fi

source "$(dirname "${BASH_SOURCE[0]}")/entrypoint_common.sh"
