#!/usr/bin/env bash
# run_experiments.sh -- the standard measurement of the float-float BK1
# kernels on a GPU node, into one results directory:
#
#   regs.txt / regs.csv   ptxas registers and spills of both CUDA kernels
#                         (and of the OpenMP kernels when NVCXX is set)
#   sweep.txt / sweep.csv GDoF/s of both CUDA kernels, and of the OpenMP
#                         kernels when BK1_FF is set, for ORDERS, REPEAT runs
#   sass.txt              instructions per kernel in the SASS
#   ncu.txt / ncu.csv     Nsight Compute statistics of both CUDA kernels for
#                         NCU_ORDERS (skipped when ncu is not on the PATH or
#                         NCU=0)
#
# Run from the repository root.  Environment:
#   OUT=results/<date>   results directory
#   ORDERS=2-8           orders p for the sweep         REPEAT=1   runs per order
#   NELMT=1000000        elements                       NTESTS=5   timed runs per kernel
#   NCU_ORDERS="7 8"     orders for ncu                 NCU_NELMT=200000
#   NVCCEXTRA=...        extra nvcc flags for the build (e.g. -DFF_CUDA_CONTRACTION_UNROLL=0)
#   BK1_FF=./BK1_ff      an OpenMP driver to include in the sweep (built separately, e.g. with nvc++)
#   NVCXX="nvc++ -mp=gpu ..."  when set, also compiles BK1_ff.cpp with -gpu=ptxinfo for the
#                        register table of the OpenMP kernels (regs_omp.txt) and with
#                        -Minfo=all for the loop messages of the plane phases (minfo.txt)
#   BK_RANDOM=1          random data (passed through to the drivers)
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cd "$root"

OUT=${OUT:-results/$(date +%Y%m%d-%H%M%S)}
ORDERS=${ORDERS:-2-8}
REPEAT=${REPEAT:-1}
NELMT=${NELMT:-1000000}
NTESTS=${NTESTS:-5}
NCU_ORDERS=${NCU_ORDERS:-7 8}
NCU_NELMT=${NCU_NELMT:-200000}
NVCCEXTRA=${NVCCEXTRA:-}
mkdir -p "$OUT"
echo "results in $OUT"
{
    echo "date: $(date)"
    echo "host: $(hostname)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "nvcc: $(${NVCC:-nvcc} --version | tail -2 | head -1)"
    nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm --format=csv 2>/dev/null || true
    echo "NVCCEXTRA: $NVCCEXTRA  BK_RANDOM: ${BK_RANDOM:-}"
} > "$OUT/env.txt"

# 1. build with the ptxas report, and tabulate it
echo "== build (ptxas -v)"
make -B BK1_ff_cuda_plain NVCCEXTRA="-Xptxas -v $NVCCEXTRA" > "$OUT/build.log" 2>&1
"$here/ptxas_table.py" --csv "$OUT/regs.csv" "$OUT/build.log" | tee "$OUT/regs.txt"
if [ -n "${NVCXX:-}" ]; then
    echo "== nvc++ ptxinfo"
    # shellcheck disable=SC2086
    $NVCXX -gpu=ptxinfo -c BK1_ff.cpp -o "$OUT/BK1_ff.o" > "$OUT/nvcxx.log" 2>&1 || true
    "$here/ptxas_table.py" --csv "$OUT/regs_omp.csv" "$OUT/nvcxx.log" | tee "$OUT/regs_omp.txt"
    echo "== nvc++ -Minfo=all (loop messages for the plane phases, BK1_ff.cpp lines 500-720)"
    # shellcheck disable=SC2086
    $NVCXX -Minfo=all -c BK1_ff.cpp -o "$OUT/BK1_ff.o" > "$OUT/minfo.log" 2>&1 || true
    awk '/^ *([5-6][0-9][0-9]|7[0-1][0-9]), /' "$OUT/minfo.log" | grep -i 'unroll\|vector\|loop' | sort -u | tee "$OUT/minfo.txt"
    rm -f "$OUT/BK1_ff.o"
fi

# 2. the sweep
echo "== sweep"
omp_arg=()
if [ -n "${BK1_FF:-}" ]; then
    omp_arg=(--omp "$BK1_FF")
fi
"$here/sweep.py" --cuda ./BK1_ff_cuda_plain "${omp_arg[@]}" --orders "$ORDERS" --nelmt "$NELMT" \
    --ntests "$NTESTS" --repeat "$REPEAT" --csv "$OUT/sweep.csv" | tee "$OUT/sweep.txt"

# 3. instructions per kernel
echo "== sass"
if command -v cuobjdump >/dev/null 2>&1; then
    "$here/sass_count.sh" ./BK1_ff_cuda_plain ${BK1_FF:+"$BK1_FF"} | tee "$OUT/sass.txt"
else
    echo "cuobjdump not found, skipped" | tee "$OUT/sass.txt"
fi

# 4. ncu
if [ "${NCU:-1}" != "0" ] && command -v ncu >/dev/null 2>&1; then
    echo "== ncu"
    # shellcheck disable=SC2086
    "$here/ncu_stats.py" --binary ./BK1_ff_cuda_plain --orders $NCU_ORDERS --nelmt "$NCU_NELMT" \
        --csv "$OUT/ncu.csv" --raw "$OUT/ncu_raw" | tee "$OUT/ncu.txt"
else
    echo "== ncu skipped (not on the PATH or NCU=0)"
fi
echo "done: $OUT"
