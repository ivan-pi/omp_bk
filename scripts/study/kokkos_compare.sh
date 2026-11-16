#!/bin/bash
# kokkos_compare.sh -- our kernels against the Kokkos CEED_BK kernels of
# dealii-X/benchmarks (scripts/study/kokkos_build.sh) at a saturating size:
# per BK and order, every variant of bk<k>_offload_<p> (OpenMP, not ref) and
# bk<k>_sycl_<p> (SYCL), and the Kokkos templated_kokkos_benchmark with its
# default element batch and with batches 1, 2, 4, ..., 32 (the ones that fit
# a work-group of 1024 and 64 KiB of SLM), REPS runs each.  BK5: only our
# 6-factor (symmetric G) variants, as Kokkos stores 6.
#
# usage:  scripts/study/kokkos_compare.sh [out.csv]
# env:    KOKKOS_BK=<dir>  build tree of CEED_BK (BK1/, BK3/, BK5/ binaries)
#         BKS="1 3 5" ORDERS="1 2 3 4 5 6 7 8" REPS=3 NTESTS=10
#         DOFS_BK1=5e7 DOFS_BK3=2e7 DOFS_BK5=5e7   (nelmt = DOFS / (p + 1)^3)
# CSV:    bk,p,impl,variant,rep,gdofs,check   (Kokkos variant b<batch>,
#         db<batch> for its default batch)
#         impl: omp, sycl or kokkos; check: our relative error (FAIL if
#         flagged) or the Kokkos output norm (equal across its batches)
. "$(dirname "$0")/../common.sh"
OUT=${1:-kokkos_compare.csv}
KOKKOS_BK=${KOKKOS_BK:?set KOKKOS_BK to the CEED_BK build tree}
BKS=${BKS:-"1 3 5"}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
REPS=${REPS:-3}
NTESTS=${NTESTS:-10}
echo "bk,p,impl,variant,rep,gdofs,check" > "$OUT"

for bk in $BKS; do
    case $bk in
        1) dofs=${DOFS_BK1:-5e7} off=2 slm_boxes=2 ;;
        3) dofs=${DOFS_BK3:-2e7} off=2 slm_boxes=4 ;;
        5) dofs=${DOFS_BK5:-5e7} off=1 slm_boxes=4 ;;
    esac
    for p in $ORDERS; do
        nq=$((p + off))
        ne=$(python3 -c "print(max(1, round($dofs / ($p + 1) ** 3)))")
        for rep in $(seq 1 "$REPS"); do
            for api in omp sycl; do
                bin=bk${bk}_offload_$p
                if [ "$api" = sycl ]; then
                    bin=bk${bk}_sycl_$p
                fi
                BK_KERNEL=-ref "$SCRIPTS/run.sh" "$bin" "$ne" "$NTESTS" 2>&1 |
                    awk -v bk="$bk" -v p="$p" -v api="$api" -v rep="$rep" '
                        $2 == "GDoF/s" && $1 !~ /^ref/ && (bk != 5 || $1 ~ /^[a-z]+6/) {
                            printf "%s,%s,%s,%s,%s,%s,%s\n", bk, p, api, $1, rep, $4,
                                   ($0 ~ /FAIL/) ? "FAIL" : $NF }' >> "$OUT"
            done
            # Kokkos: argv = p nelmt [nelmtPerBatch]; the default batch first
            for b in default 1 2 4 8 16 32; do
                args=("$p" "$ne")
                if [ "$b" != default ]; then
                    if [ $((nq * nq * b)) -gt 1024 ] || [ $((4 * slm_boxes * b * nq * nq * nq)) -gt 61440 ]; then
                        continue
                    fi
                    args+=("$b")
                fi
                BUILD_DIR=$KOKKOS_BK/BK$bk "$SCRIPTS/run.sh" templated_kokkos_benchmark "${args[@]}" 2>&1 |
                    awk -v bk="$bk" -v p="$p" -v rep="$rep" -v tag="$([ "$b" = default ] && echo d)" '
                        $1 == "BK" bk && NF >= 11 {
                            printf "%s,%s,kokkos,%sb%s,%s,%s,%s\n", bk, p, tag, $4, rep, $9, $11 }' >> "$OUT"
            done
        done
        echo "BK$bk p=$p done" >&2
    done
done

# best variant (median over the repetitions) per BK, order and impl
python3 - "$OUT" <<'PY'
import csv, statistics, sys
from collections import defaultdict
g = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if r["check"] != "FAIL":
        g[(int(r["bk"]), int(r["p"]), r["impl"], r["variant"])].append(float(r["gdofs"]))
best = {}
for (bk, p, impl, v), x in g.items():
    m = statistics.median(x)
    if best.get((bk, p, impl), ("", 0))[1] < m:
        best[(bk, p, impl)] = (v, m)
for bk, p in sorted({k[:2] for k in best}):
    row = {i: best.get((bk, p, i), ("-", float("nan"))) for i in ("kokkos", "omp", "sycl")}
    ours = max(row["omp"][1], row["sycl"][1])
    print(f"BK{bk} p={p}  kokkos {row['kokkos'][1]:6.2f} ({row['kokkos'][0]:4s})  "
          f"omp {row['omp'][1]:6.2f} ({row['omp'][0]:10s})  sycl {row['sycl'][1]:6.2f} ({row['sycl'][0]:14s})  "
          f"ours/kokkos {ours / row['kokkos'][1]:4.2f}")
PY
