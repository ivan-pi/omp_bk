#!/bin/bash
# bk5_study.sh -- every variant of openmp/offload/bk5_offload.cpp (symmetric half vs full
# geometric tensor, two layouts, reference vs per-point kernel) per order at
# DOFS DoFs, REPS runs of NTESTS launches.
# CSV: nq,p,variant,F,rep,gdofs,gbs,relerr
. "$(dirname "$0")/../common.sh"
OUT=${1:-bk5_study.csv}
NQS=${NQS:-"2 3 4 5 6 7 8 9"}
DOFS=${DOFS:-5e7}
REPS=${REPS:-3}
NTESTS=${NTESTS:-5}
echo "nq,p,variant,F,rep,gdofs,gbs,relerr" > "$OUT"
for nq in $NQS; do
    ne=$(python3 -c "print(max(1, round($DOFS / $nq ** 3)))")
    for rep in $(seq 1 "$REPS"); do
        "$SCRIPTS/run.sh" "bk5_offload_$nq" "$ne" "$NTESTS" 2>&1 | grep -E "GDoF/s" | \
            awk -v nq="$nq" -v rep="$rep" '{F = ($1 ~ /9/) ? 9 : 6; err = ($NF == "FAIL") ? "FAIL" : $(NF);
                 printf "%d,%d,%s,%d,%d,%s,%s,%s\n", nq, nq - 1, $1, F, rep, $4, $7, err}' >> "$OUT"
    done
    echo "nq=$nq done"
done
