#!/bin/bash
# run.sh -- run a binary of $BUILD_DIR (scripts/build.sh) on the GPU.
#
# usage:  scripts/run.sh <name> [args...]     e.g.  run.sh BK1_4 4 524288 10
#                                                       run.sh stream_omp
#
# Offload is mandatory (no silent host fallback), the device is
# $ONEAPI_DEVICE_SELECTOR (default level_zero:0) and the run is memory-capped
# (common.sh).  BK1_ff binaries run with the driver's FP64 emulation
# (IGC_EnableDPEmulation=1, OverrideDefaultFP64Settings=1).  PROFILE=1 adds
# the Level Zero plugin's per-kernel host and device times
# (LIBOMPTARGET_PLUGIN_PROFILE=T).  The last line on stderr is the peak memory.

. "$(dirname "$0")/common.sh"
name=$1
shift
if [ ! -x "$BUILD_DIR/$name" ]; then
    echo "no binary $BUILD_DIR/$name; build it with scripts/build.sh $name" >&2
    exit 2
fi

extra=()
case $name in
    BK1_ff*) extra+=(IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1) ;;
esac
if [ "${PROFILE:-0}" = 1 ]; then
    extra+=(LIBOMPTARGET_PLUGIN_PROFILE=T)
fi

args=("$@")
oneapi_env
capped env OMP_TARGET_OFFLOAD=MANDATORY "${extra[@]}" "$BUILD_DIR/$name" "${args[@]}"
