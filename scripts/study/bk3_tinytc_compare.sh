#!/bin/bash
# bk3_tinytc_compare.sh -- BK3 in tinytc against our OpenMP and SYCL kernels
# at one size: every variant of bk3_offload_<p> and bk3_sycl_<p> (OpenMP not
# ref; scripts/study/bk3_pick.sh) and the tinytc configurations of
# picks.txt (scripts/study/tinytc_final.sh), REPS runs each.
#
# usage:  scripts/study/bk3_tinytc_compare.sh <picks.txt> [out.csv]
#         picks.txt: lines "3 <layout> <p> <wgs> <sg> <B>" (tinytc_final.sh)
# env:    ORDERS="1 ... 8" DOFS=2e7 REPS=3 NTESTS=10
# CSV:    bk,p,impl,variant,rep,gdofs,check   (scripts/plot/plot_compare.py;
#         impl: omp, sycl, tinytc (blocked layout: blk, blk_ch) or tinytc_ec
#         (ours); tinytc variant <layout>_sg<sg>_b<B>_w<wgs>, layout blk_gs or
#         ec_gs where S spills to global memory)
. "$(dirname "$0")/../common.sh"
PICKS=$1
OUT=${2:-bk3_tinytc_compare.csv}
export DOFS=${DOFS:-2e7} REPS=${REPS:-3} NTESTS=${NTESTS:-10}
ours=$(mktemp)
tt=$(mktemp)
"$SCRIPTS/study/bk3_pick.sh" "$ours" >&2
"$SCRIPTS/study/tinytc_final.sh" "$PICKS" "$tt" > /dev/null
{
    echo "bk,p,impl,variant,rep,gdofs,check"
    awk -F, 'NR > 1 { printf "3,%s,%s,%s,%s,%s,%s\n", $2, $1, $3, $4, $5, $7 }' "$ours"
    awk -F, 'NR > 1 { printf "3,%s,%s,%s_sg%s_b%s_w%s,%s,%s,%s\n", $3, ($2 ~ /^ec/) ? "tinytc_ec" : "tinytc",
                      $2, $15, $16, $5, $6, $9,
                      ($14 == "ok") ? $13 : "FAIL" }' "$tt"
} > "$OUT"
rm -f "$ours" "$tt"
echo "wrote $OUT" >&2
