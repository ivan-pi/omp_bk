#!/bin/bash
# build.sh -- the reproducer for nq = 3 and nq = 9, ahead of time for the A770
# (acm-g10), with icpx from oneAPI; IGC's assembly of both kernels goes to
# asm_<nq>/ (IGC_ShaderDumpEnable).  FLAGS: more compiler flags (default
# -fp-model=precise, as the benchmark builds use).
set -e
cd "$(dirname "$0")"
. /opt/intel/oneapi/compiler/latest/env/vars.sh > /dev/null 2>&1
DEVICE=${DEVICE:-acm-g10}
FLAGS=${FLAGS--fp-model=precise}
for nq in 3 9; do
    rm -rf "asm_$nq"
    IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$PWD/asm_$nq" \
        icpx -fsycl -O3 $FLAGS -DNQ=$nq -fsycl-targets=spir64_gen \
        -Xsycl-target-backend=spir64_gen "-device $DEVICE" repro.cpp -o "repro_$nq"
done
