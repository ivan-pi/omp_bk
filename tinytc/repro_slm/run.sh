#!/bin/bash
# run.sh -- run the reproducer in the oneAPI environment: ./run.sh <N> [<N> ...]
# (the arguments are saved under a name setvars.sh does not overwrite)
cd "$(dirname "$0")"
repro_args=("$@")
set --
. /opt/intel/oneapi/setvars.sh > /dev/null
exec ./repro "${repro_args[@]}"
