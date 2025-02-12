#!/bin/bash

if [[ -z "${@}" ]]; then
    exec /bin/bash
fi

eval exec modelconverter "${@}"
