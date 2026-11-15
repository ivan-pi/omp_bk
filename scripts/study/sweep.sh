#!/bin/bash
# sweep.sh -- build (AOT, one order per binary) and run BK1, BK3, BK5 and the
# STREAM test on the GPU; one CSV row per program and order.
#
# usage:  scripts/study/sweep.sh [results.csv]
# env:    ORDERS="2 4 6"     polynomial orders p (BK1, BK3: nq = p + 2; BK5: nq = p + 1)
#         NELMT=524288       elements (host memory grows as nelmt * nq^3:
#                            BK3 needs ~7.5 GB at p = 6, ~11 GB at p = 7)
#         NTESTS=10          repetitions (the driver reports the minimum)
#         PROGS="BK1 BK3 BK5"
#         BK1_ff is checked and timed separately (see README); add it to
#         PROGS to include its float row in the CSV.
#
# Builds run one at a time and never alongside a run.  CSV columns:
#   prog,p,arg,nelmt,gdofs,gbs,peak_mb   (arg = the driver's first argument, p)
# The raw output of each run is kept in $BUILD_DIR/<prog>_<p>.out.

. "$(dirname "$0")/../common.sh"
OUT=${1:-gpu_results.csv}
ORDERS=${ORDERS:-"2 4 6"}
NELMT=${NELMT:-524288}
NTESTS=${NTESTS:-10}
PROGS=${PROGS:-"BK1 BK3 BK5"}

targets=(stream_omp)
for prog in $PROGS; do
    for p in $ORDERS; do
        targets+=("${prog}_$p")
    done
done
"$SCRIPTS/build.sh" "${targets[@]}" || exit 1

echo "prog,p,arg,nelmt,gdofs,gbs,peak_mb" > "$OUT"
for prog in $PROGS; do
    for p in $ORDERS; do
        log=$BUILD_DIR/${prog}_$p.out
        "$SCRIPTS/run.sh" "${prog}_$p" "$p" "$NELMT" "$NTESTS" > "$log" 2>&1
        # BK1/BK3/BK5 print "GDoF/s = x GB/s = y"; BK1_ff its float row
        line=$(grep -E "^(SumFactorization|float ) .*GDoF/s" "$log" | head -1)
        g=$(echo "$line" | sed -n 's/.*GDoF\/s = *\([^ ]*\) GB\/s = *\([^ ]*\).*/\1,\2/p')
        pk=$(sed -n 's/^peak=\([0-9]*\)MB/\1/p' "$log" | tail -1)
        echo "$prog,$p,$p,$NELMT,${g:-NaN,NaN},${pk:-NaN}" | tee -a "$OUT"
    done
done
"$SCRIPTS/run.sh" stream_omp > "$BUILD_DIR/stream_omp.out" 2>&1
grep -E "^(read|write|copy|scale|add|triad|bk1p2|check)" "$BUILD_DIR/stream_omp.out"
