#!/bin/bash
# compare_tips.sh -- the oneAPI-guide variants (SLM padding, power-of-two
# "fill" groups, SYCL sub-group block copies) against the current best
# kernels, per order, at DOFS (default 1e8), REPS runs of NTESTS launches.
# env: ORDERS="1 2 3 4 5 6 7 8" (p, nq = p + 2)
# CSV: api,nq,p,variant,rep,gdofs,gdofs_dev,check
. "$(dirname "$0")/../common.sh"
OUT=${1:-compare_tips.csv}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
DOFS=${DOFS:-1e8}
REPS=${REPS:-3}
NTESTS=${NTESTS:-5}
OMP="gsI_b1 gsI_b4 gsI_b8 gsI_b16 gsT_b1 gsT_b4 gsT_b8 gsW_b2 gsD1 gsP_b1 gsP_b4 gsP_b8 gsP_b16 gsF_f128 gsF_f256 gsF_f512 gsFP_f128 gsFP_f256 gsFP_f512"
SYCL="elem inl_b1 inl_b4 inl_b8 inl_b16 inlP_b1 inlP_b4 inlP_b8 inlP_b16 inlB_b1 inlB_b4 inlB_b16 inlBP_b1 inlBP_b4 inlBP_b16 inlF_f128 inlF_f256 inlF_f512 inlFP_f128 inlFP_f256 inlFP_f512"
echo "api,nq,p,variant,rep,gdofs,gdofs_dev,check" > "$OUT"
for p in $ORDERS; do
    nq=$((p + 2))
    ne=$(python3 -c "print(max(1, round($DOFS / ($nq - 1) ** 3)))")
    ref=0
    omp=$OMP
    if [ "$p" -le 2 ]; then
        ref=1   # the reference kernel is a contender (p = 1) and the check
    fi
    for rep in $(seq 1 "$REPS"); do
        BK_REF=$ref BK_KERNEL="$omp" "$SCRIPTS/run.sh" "bk1_offload_$p" "$ne" "$NTESTS" 2>&1 | grep -E "GDoF/s" | \
            awk -v nq="$nq" -v rep="$rep" '{chk = ($NF == "(reference)") ? "reference" : $NF;
                 printf "omp,%d,%d,%s,%d,%s,,%s\n", nq, nq - 2, $1, rep, $4, chk}' >> "$OUT"
        BK_KERNEL="$SYCL" "$SCRIPTS/run.sh" "bk1_sycl_$p" "$ne" "$NTESTS" 2>&1 | grep -E "GDoF/s" | sed 's/(device: GDoF\/s =//' | \
            awk -v nq="$nq" -v rep="$rep" '{chk = ($NF == "FAIL") ? "FAIL" : $(NF);
                 printf "sycl,%d,%d,%s,%d,%s,%s,%s\n", nq, nq - 2, $1, rep, $4, $8, chk}' >> "$OUT"
    done
    echo "p=$p done"
done
