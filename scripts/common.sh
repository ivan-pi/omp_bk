# common.sh -- sourced by the Intel GPU scripts (scripts/ and its study/,
# profile/ subdirectories): oneAPI environment, paths and the memory-capped
# scope every run goes through.
#
# env:  ONEAPI_ROOT=/opt/intel/oneapi   oneAPI install
#       ICPX_VERSION=latest             compiler version under $ONEAPI_ROOT/compiler
#       GPU_DEVICE=acm-g10              ocloc AOT target (acm-g10 = Arc A770/A750, DG2)
#       ONEAPI_DEVICE_SELECTOR=level_zero:0
#       MEM_MAX=8G                      memory cap per run (systemd scope)
#       BUILD_DIR=<repo>/build/a770     CMake build tree of the binaries (preset a770)

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPTS/.." && pwd)
BUILD_DIR=$(realpath -m "${BUILD_DIR:-$REPO/build/a770}")
ONEAPI_ROOT=${ONEAPI_ROOT:-/opt/intel/oneapi}
ICPX_VERSION=${ICPX_VERSION:-latest}
GPU_DEVICE=${GPU_DEVICE:-acm-g10}
MEM_MAX=${MEM_MAX:-8G}
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}

# The oneAPI vars.sh scripts parse (and consume) the positional parameters,
# so they are sourced from a function with an empty argument list.
oneapi_env() {
    set --
    . "$ONEAPI_ROOT/compiler/$ICPX_VERSION/env/vars.sh" > /dev/null 2>&1
    # The Level Zero and OpenCL adapters of the 2025+ runtime need libumf;
    # without it sycl-ls and the OpenMP plugin find no device.
    if [ -f "$ONEAPI_ROOT/umf/latest/env/vars.sh" ]; then
        . "$ONEAPI_ROOT/umf/latest/env/vars.sh" > /dev/null 2>&1
    fi
    if [ -f "$ONEAPI_ROOT/tbb/latest/env/vars.sh" ]; then
        . "$ONEAPI_ROOT/tbb/latest/env/vars.sh" > /dev/null 2>&1
    fi
}

# capped <cmd...>: run cmd in a systemd user scope limited to $MEM_MAX (no
# swap), so a too-large problem fails on its own instead of driving the
# machine into the OOM killer; prints "peak=<MB>MB" on stderr afterwards.
# Without systemd-run the command runs uncapped.
capped() {
    if command -v systemd-run > /dev/null 2>&1; then
        systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
            bash -c 'cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
                     "$@"; rc=$?
                     [ -r $cg/memory.peak ] && echo "peak=$(( $(cat $cg/memory.peak) / 1048576 ))MB" >&2
                     grep -q "oom_kill [1-9]" $cg/memory.events 2>/dev/null && echo "OOM: killed at MemoryMax=$MEM_MAX" >&2
                     exit $rc' capped "$@"
    else
        "$@"
    fi
}
export MEM_MAX
