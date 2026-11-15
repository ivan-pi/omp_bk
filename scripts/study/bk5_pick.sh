#!/bin/bash
# bk5_pick.sh -- the fastest BK5 variant per API, order and tensor storage at
# a fixed size: every variant of build/a770/bk5_offload_<nq> (OpenMP: pt, sf, cu;
# not ref, tid, fill) and build/a770/bk5_sycl_<nq> (SYCL: sf, not _sg16), REPS
# runs each, median host GDoF/s.  Prints the KERNELS list for bk5_sweep.sh.
#
# usage:  scripts/study/bk5_pick.sh [out.csv]
# env:    NQS="2 3 4 5 6 7 8 9" DOFS=5e7 REPS=3 NTESTS=10
# CSV:    api,nq,variant,rep,gdofs     (api = <omp|sycl><6|9>)
. "$(dirname "$0")/../common.sh"
OUT=${1:-bk5_pick.csv}
NQS=${NQS:-"2 3 4 5 6 7 8 9"}
DOFS=${DOFS:-5e7}
REPS=${REPS:-3}
NTESTS=${NTESTS:-10}
echo "api,nq,variant,rep,gdofs" > "$OUT"
for nq in $NQS; do
    ne=$(python3 -c "print(max(1, round($DOFS / $nq ** 3)))")
    for lang in omp sycl; do
        bin=bk5_offload_$nq
        if [ "$lang" = sycl ]; then
            bin=bk5_sycl_$nq
        fi
        for rep in $(seq 1 "$REPS"); do
            "$SCRIPTS/run.sh" "$bin" "$ne" "$NTESTS" 2>&1 |
                awk -v lang="$lang" -v nq="$nq" -v rep="$rep" '
                    $2 == "GDoF/s" && $0 !~ /FAIL/ && $1 ~ /^(pt|sf|cu)/ && $1 !~ /_sg16|_f[0-9]/ {
                        F = ($1 ~ /^(pt|sf|cu)9/) ? 9 : 6
                        printf "%s%d,%s,%s,%s,%s\n", lang, F, nq, $1, rep, $4
                    }' >> "$OUT"
        done
    done
done
python3 - "$OUT" <<'PY'
import csv, statistics, sys
from collections import defaultdict
runs = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    runs[(r["api"], int(r["nq"]), r["variant"])].append(float(r["gdofs"]))
best = {}
for (api, nq, v), g in runs.items():
    m = statistics.median(g)
    if (api, nq) not in best or m > best[(api, nq)][1]:
        best[(api, nq)] = (v, m)
for (api, nq), (v, m) in sorted(best.items(), key=lambda x: (x[0][0][-1], x[0][0], x[0][1])):
    print(f"{api:6s} p={nq - 1} {v:16s} {m:7.3f} GDoF/s")
print("KERNELS=\"" + " ".join(f"{api[:-1]}:{nq}:{v}" for (api, nq), (v, m) in
                              sorted(best.items(), key=lambda x: (x[0][0][-1], x[0][0], x[0][1]))) + "\"")
PY
