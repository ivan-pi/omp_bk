#!/bin/bash
# bk3_pick.sh -- the fastest BK3 variant per API and order at a fixed size:
# every variant of build/a770/bk3_offload_<p> (OpenMP, not ref) and
# build/a770/bk3_sycl_<p> (SYCL), REPS runs each, median host GDoF/s.
#
# usage:  scripts/study/bk3_pick.sh [out.csv]
# env:    ORDERS="1 2 3 4 5 6 7 8" (p, nq = p + 2) DOFS=2e7 REPS=3 NTESTS=10
#         (G is 6 nq^3 floats per element: ~1.6 GB of host and device
#         memory at p = 1 and 2e7 DoFs)
#         COPY_GBS, READ9_GBS: the roofs (default: one run of stream_omp,
#         copy and read9 = BK3's mix of ~90% reads)
# CSV:    api,p,variant,rep,gdofs,gbs,relerr     (api: omp or sycl)
. "$(dirname "$0")/../common.sh"
OUT=${1:-bk3_pick.csv}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
DOFS=${DOFS:-2e7}
REPS=${REPS:-3}
NTESTS=${NTESTS:-10}
echo "api,p,variant,rep,gdofs,gbs,relerr" > "$OUT"
for p in $ORDERS; do
    ne=$(python3 -c "print(max(1, round($DOFS / ($p + 1) ** 3)))")
    for bin in bk3_offload_$p bk3_sycl_$p; do
        api=omp
        if [ "$bin" = bk3_sycl_$p ]; then
            api=sycl
        fi
        for rep in $(seq 1 "$REPS"); do
            # $NF: the relative error, or FAIL after it
            BK_KERNEL=-ref "$SCRIPTS/run.sh" "$bin" "$ne" "$NTESTS" 2>&1 |
                awk -v api="$api" -v p="$p" -v rep="$rep" \
                    '$2 == "GDoF/s" { printf "%s,%s,%s,%s,%s,%s,%s\n", api, p, $1, rep, $4, $7, $NF }' >> "$OUT"
        done
    done
    echo "p=$p done" >&2
done
if [ -z "$COPY_GBS" ] || [ -z "$READ9_GBS" ]; then
    stream=$("$SCRIPTS/run.sh" stream_omp 2>&1)
    COPY_GBS=${COPY_GBS:-$(echo "$stream" | awk '$1 == "copy" {print $NF}')}
    READ9_GBS=${READ9_GBS:-$(echo "$stream" | awk '$1 == "read9" {print $NF}')}
fi
python3 - "$OUT" "$COPY_GBS" "$READ9_GBS" <<'PY'
import csv, statistics, sys
from collections import defaultdict
gdofs = defaultdict(list)
gbs = defaultdict(list)
bad = set()
for r in csv.DictReader(open(sys.argv[1])):
    k = (r["api"], int(r["p"]), r["variant"])
    gdofs[k].append(float(r["gdofs"]))
    gbs[k].append(float(r["gbs"]))
    if r["relerr"] == "FAIL":
        bad.add(k)
best = {}
for k, g in gdofs.items():
    if k not in bad and ((k[0], k[1]) not in best or statistics.median(g) > best[(k[0], k[1])][1]):
        best[(k[0], k[1])] = (k[2], statistics.median(g), statistics.median(gbs[k]))
copy, read9 = float(sys.argv[2]), float(sys.argv[3])
print(f"roofs: STREAM copy {copy:.0f} GB/s, read9 (90% reads) {read9:.0f} GB/s")
for (api, p), (v, g, b) in sorted(best.items()):
    print(f"{api:5s} p={p} {v:12s} {g:6.2f} GDoF/s  {b:5.0f} GB/s  ({100 * b / copy:3.0f}% of copy, {100 * b / read9:3.0f}% of read9)")
if bad:
    print("FAIL:", sorted(bad))
PY
