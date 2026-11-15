#!/bin/bash
# build_aot.sh -- ahead-of-time build for an Intel GPU with icpx, one
# polynomial order per binary.
#
# usage:  gpu/scripts/build_aot.sh <BK1|BK3|BK5|BK1_ff> <order>   -> $BUILD_DIR/<prog>_<order>
#         gpu/scripts/build_aot.sh stream_omp                     -> $BUILD_DIR/stream_omp
#   order is the driver's first argument: p for BK1, BK3, BK1_ff; nq for BK5.
#
# Each driver instantiates every order in a switch; compiled as is, the device
# image holds all of them.  Here a copy of the driver keeps only the requested
# case, so the device compiler (ocloc/IGC) sees one kernel set at a time.  The
# compile runs at low priority in a memory-capped scope (common.sh) and prints
# its peak memory.  BK1_ff is built with IGC_EnableDPEmulation=1: the A770
# has no fp64 hardware, and the double reference and the split/join of the
# float-float kernels need it.
#
# env:  see common.sh (GPU_DEVICE, MEM_MAX, BUILD_DIR, ICPX_VERSION);
#       CXXFLAGS_GPU overrides the flags below.

set -e
. "$(dirname "$0")/common.sh"
prog=$1
order=$2

case $prog in
    BK1|BK3|BK5|BK1_ff)
        [ -n "$order" ] || { echo "usage: $0 $prog <order>" >&2; exit 2; }
        name=${prog}_$order
        mkdir -p "$BUILD_DIR"
        src=$BUILD_DIR/$name.cpp
        sed -E "/^ *case [0-9]+:/{/^ *case $order:/!d}" "$REPO/$prog.cpp" > "$src"
        if ! grep -q -E "^ *case $order:" "$src"; then
            echo "$prog has no case $order" >&2
            exit 2
        fi
        ;;
    stream_omp)
        name=stream_omp
        mkdir -p "$BUILD_DIR"
        src=$REPO/gpu/stream_omp.cpp
        ;;
    *)
        echo "usage: $0 <BK1|BK3|BK5|BK1_ff> <order> | stream_omp" >&2
        exit 2
        ;;
esac

# -fp-model=precise: icpx defaults to a fast model, which breaks the
# error-free transformations of float_float.h.
CXXFLAGS_GPU=${CXXFLAGS_GPU:-"-Wall -O3 -std=c++17 -march=native -fp-model=precise -fiopenmp -fopenmp-targets=spir64_gen"}
dp_emu=""
if [ "$prog" = BK1_ff ]; then
    dp_emu="IGC_EnableDPEmulation=1"
fi

oneapi_env
start=$(date +%s)
set +e
capped env $dp_emu nice -n 19 icpx $CXXFLAGS_GPU -I"$REPO" \
    -Xopenmp-target-backend=spir64_gen "-device $GPU_DEVICE" \
    -o "$BUILD_DIR/$name" "$src" > "$BUILD_DIR/$name.build.log" 2>&1
rc=$?
set -e
peak=$(grep -o "peak=[0-9]*MB" "$BUILD_DIR/$name.build.log" | tail -1)
echo "$name rc=$rc $peak time=$(( $(date +%s) - start ))s"
if [ $rc -ne 0 ]; then
    tail -20 "$BUILD_DIR/$name.build.log" >&2
fi
exit $rc
