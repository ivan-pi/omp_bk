#!/bin/bash
# bk3_pick.sh -- the fastest BK3 variant per API and order at a fixed size:
# every variant of build/a770/bk3_offload_<p> (OpenMP, not ref) and
# build/a770/bk3_sycl_<p> (SYCL), REPS runs each, median host GDoF/s.
#
# usage:  scripts/study/bk3_pick.sh [out.csv]
# env:    ORDERS="1 2 3 4 5 6 7 8" (p, nq = p + 2) DOFS=2e7 REPS=3 NTESTS=10
#         COPY_GBS=459 READ9_GBS (default: stream_omp's read9, BK3's mix of
#         ~90% reads) -- the two bandwidth roofs the summary reports against
#         (G is 6 nq^3 floats per element: ~1.6 GB of host and device
#         memory at p = 1 and 2e7 DoFs)
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
    for api in omp sycl; do
        bin=bk3_offload_$p
        if [ "$api" = sycl ]; then
            bin=bk3_sycl_$p
        fi
        for rep in $(seq 1 "$REPS"); do
            "$SCRIPTS/run.sh" "$bin" "$ne" "$NTESTS" 2>&1 |
                awk -v api="$api" -v p="$p" -v rep="$rep" '
                    $2 == "GDoF/s" && $1 != "ref" {
                        err = ($NF == "FAIL") ? "FAIL" : $NF
                        for (i = 1; i <= NF; ++i) { if ($i == "relerr") { err = ($NF == "FAIL") ? "FAIL" : $(i + 2) } }
                        printf "%s,%s,%s,%s,%s,%s,%s\n", api, p, $1, rep, $4, $7, err
                    }' >> "$OUT"
        done
    done
    echo "p=$p done" >&2
done
COPY_GBS=${COPY_GBS:-459}
if [ -z "$READ9_GBS" ]; then
    READ9_GBS=$("$SCRIPTS/run.sh" stream_omp 2>&1 | awk '$1 == "read9" {print $NF}')
fi
python3 - "$OUT" "$COPY_GBS" "${READ9_GBS:-0}" <<'PY'
import csv, statistics, sys
from collections import defaultdict
runs = defaultdict(list)
bad = set()
for r in csv.DictReader(open(sys.argv[1])):
    k = (r["api"], int(r["p"]), r["variant"])
    runs[k].append(float(r["gdofs"]))
    if r["relerr"] == "FAIL":
        bad.add(k)
best = {}
for (api, p, v), g in runs.items():
    if (api, p, v) in bad:
        continue
    m = statistics.median(g)
    if (api, p) not in best or m > best[(api, p)][1]:
        best[(api, p)] = (v, m)
copy, read9 = float(sys.argv[2]), float(sys.argv[3])
print(f"roofs: STREAM copy {copy:.0f} GB/s, read9 (90% reads) {read9:.0f} GB/s")
for (api, p), (v, m) in sorted(best.items(), key=lambda x: (x[0][0], x[0][1])):
    nm, nq = p + 1, p + 2
    bpd = 4.0 * (2 * nm ** 3 + 6 * nq ** 3) / nm ** 3   # bytes per DoF: in + out + G
    r9 = f", {100 * m * bpd / read9:3.0f}% of read9" if read9 > 0 else ""
    print(f"{api:5s} p={p} {v:14s} {m:6.2f} GDoF/s  {m * bpd:5.0f} GB/s  ({100 * m * bpd / copy:3.0f}% of copy{r9})")
if bad:
    print("FAIL:", sorted(bad))
PY
