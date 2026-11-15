#!/bin/bash
# build.sh -- build Intel GPU targets with CMake (preset a770: icpx, ahead of
# time for $GPU_DEVICE), one target after another.
#
# usage:  scripts/build.sh <target>...              -> $BUILD_DIR/<target>
#   BK1_<p> BK3_<p> BK1_ff_<p> BK5_<nq>     openmp/ drivers, one order each
#   bk1_offload_<nq> bk5_offload_<nq>       openmp/offload/ kernel variants
#   bk1_sycl_<nq> bk5_sycl_<nq>             sycl/ (bk1_sycl_<nq>_jit: built for
#                                           JIT compilation, which the
#                                           specialization-constant variant
#                                           needs to fold nq)
#   stream_omp                              roofline/ bandwidth test
#   bk_tinytc                               tinytc/ driver (tinytc in /opt/tinytc/latest)
#   all_<prog>                              every order of <prog>
# e.g.    scripts/build.sh BK1_4 bk5_offload_7 all_bk1_sycl
#
# The first call configures $BUILD_DIR with the preset; later calls reuse it.
# Each compile and link runs at low priority in a memory-capped scope
# (scripts/capped, common.sh), never two at a time (BK_GPU_JOBS=1).  Prints
# one line per target with its exit status, largest peak memory and time;
# the full output is in $BUILD_DIR/logs/<target>.log.
#
# env:  see common.sh (GPU_DEVICE, MEM_MAX, BUILD_DIR, ICPX_VERSION);
#       EXTRA_FLAGS are added to the compile and link flags of every GPU
#       target of $BUILD_DIR (e.g. EXTRA_FLAGS="-ftarget-register-alloc-mode=..."),
#       so give such builds their own tree: BUILD_DIR=build/a770-grf.
#       Setting or clearing EXTRA_FLAGS for an existing tree rebuilds its
#       targets with the new flags.

set -e
. "$(dirname "$0")/common.sh"
if [ $# -eq 0 ]; then
    sed -n '4,/^# e.g./p' "$0" >&2
    exit 2
fi

oneapi_env
mkdir -p "$BUILD_DIR/logs"
cd "$REPO"
if ! cmake --preset a770 -B "$BUILD_DIR" -DBK_GPU_DEVICE="$GPU_DEVICE" \
        -DBK_EXTRA_FLAGS="${EXTRA_FLAGS:-}" > "$BUILD_DIR/logs/configure.log" 2>&1; then
    cat "$BUILD_DIR/logs/configure.log" >&2
    exit 1
fi

status=0
for target in "$@"; do
    log=$BUILD_DIR/logs/$target.log
    start=$(date +%s)
    set +e
    cmake --build "$BUILD_DIR" --target "$target" > "$log" 2>&1
    rc=$?
    set -e
    peak=$(sed -n 's/.*peak=\([0-9]*\)MB.*/\1/p' "$log" | sort -n | tail -1)
    echo "$target rc=$rc${peak:+ peak=${peak}MB} time=$(( $(date +%s) - start ))s"
    if [ $rc -ne 0 ]; then
        tail -20 "$log" >&2
        status=1
    fi
done
exit $status
