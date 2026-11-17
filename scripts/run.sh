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
# The run waits for the GPU to be free of other processes and is repeated
# when one shows up during it (common.sh: gpu_wait_idle, gpu_others).
# JIT-compiled kernels go to a persistent cache per build tree,
# $BUILD_DIR/sycl_cache, so only the first run of a kernel (or of a
# specialization-constant value) compiles it (SYCL_CACHE_PERSISTENT=0:
# compile every time; SYCL_CACHE_DIR: another cache).

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
extra+=(SYCL_CACHE_PERSISTENT="${SYCL_CACHE_PERSISTENT:-1}" SYCL_CACHE_DIR="${SYCL_CACHE_DIR:-$BUILD_DIR/sycl_cache}")
if [ "${PROFILE:-0}" = 1 ]; then
    extra+=(LIBOMPTARGET_PLUGIN_PROFILE=T)
fi

oneapi_env
# Other processes on the GPU (another session's benchmarks) skew the
# timings: wait until the GPU is free, watch it during the run and repeat
# the run if another process appeared (GPU_GUARD=0: neither).
for attempt in 1 2 3 4 5; do
    if [ "${GPU_GUARD:-1}" = 0 ]; then
        capped env OMP_TARGET_OFFLOAD=MANDATORY "${extra[@]}" "$BUILD_DIR/$name" "$@"
        exit
    fi
    gpu_wait_idle
    out=$(mktemp)
    capped env OMP_TARGET_OFFLOAD=MANDATORY "${extra[@]}" "$BUILD_DIR/$name" "$@" > "$out" 2>&1 &
    run=$!
    shared=""
    while kill -0 "$run" 2> /dev/null; do
        shared=${shared:-$(gpu_others $$)}
        sleep 0.5
    done
    wait "$run"
    rc=$?
    if [ -z "$shared" ]; then
        cat "$out"
        rm -f "$out"
        exit $rc
    fi
    rm -f "$out"
    echo "run.sh: the GPU was shared with $(echo $shared) during the run; repeating it" >&2
done
echo "run.sh: the GPU stayed shared; giving up" >&2
exit 1
