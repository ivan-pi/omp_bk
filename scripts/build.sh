#!/bin/bash
# build.sh -- build Intel GPU targets with CMake (preset a770: icpx, ahead of
# time for $GPU_DEVICE), one target after another.
#
# usage:  scripts/build.sh <target>...              -> $BUILD_DIR/<target>
#   BK1_<p> BK3_<p> BK1_ff_<p> BK5_<p>      openmp/ drivers, one order each
#   bk1_offload_<p> bk3_offload_<p> bk5_offload_<p>   openmp/offload/ kernel variants
#   bk3_sycl_<p>                            sycl/ BK3 kernel variants
#   bk1_sycl_<p> bk5_sycl_<p>               sycl/ (bk1_sycl_<p>_jit: built for
#                                           JIT compilation, which the
#                                           specialization-constant variant
#                                           needs to fold nq)
#   bk1_sycl_sc                             sycl/ BK1, every order in one JIT
#                                           binary, shape from a specialization
#                                           constant (run.sh bk1_sycl_sc <p>)
#   stream_omp                              roofline/ bandwidth test
#   bk_tinytc                               tinytc/ driver (TINYTC_ROOT of preset a770)
#   all_<prog>                              every order of <prog>
# e.g.    scripts/build.sh BK1_4 bk5_offload_6 all_bk1_sycl
# (every target is named by the polynomial order p)
#
# The first call configures $BUILD_DIR with the preset; later calls reuse it.
# GPU compiles and links never run two at a time (BK_GPU_JOBS=1).  Prints one
# line per target with its exit status and time; the full output is in
# $BUILD_DIR/logs/<target>.log.
#
# env:  see common.sh (GPU_DEVICE, BUILD_DIR, ICPX_VERSION);
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
# configure when the tree is new or its device or flags differ, or when it
# still has the compiler launcher of an older preset (scripts/capped)
cached() { sed -n "s/^$1:[A-Z]*=//p" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; }
if [ "$(cached BK_GPU_DEVICE)" != "$GPU_DEVICE" ] || [ "$(cached BK_EXTRA_FLAGS)" != "${EXTRA_FLAGS:-}" ] ||
        [ -n "$(cached CMAKE_CXX_COMPILER_LAUNCHER)" ]; then
    if ! cmake --preset a770 -B "$BUILD_DIR" -DBK_GPU_DEVICE="$GPU_DEVICE" \
            -UCMAKE_CXX_COMPILER_LAUNCHER -UCMAKE_CXX_LINKER_LAUNCHER \
            -DBK_EXTRA_FLAGS="${EXTRA_FLAGS:-}" > "$BUILD_DIR/logs/configure.log" 2>&1; then
        cat "$BUILD_DIR/logs/configure.log" >&2
        exit 1
    fi
fi

status=0
for target in "$@"; do
    log=$BUILD_DIR/logs/$target.log
    start=$(date +%s)
    set +e
    cmake --build "$BUILD_DIR" --target "$target" > "$log" 2>&1
    rc=$?
    set -e
    echo "$target rc=$rc time=$(( $(date +%s) - start ))s"
    if [ $rc -ne 0 ]; then
        tail -20 "$log" >&2
        status=1
    fi
done
exit $status
