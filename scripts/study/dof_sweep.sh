#!/bin/bash
# dof_sweep.sh -- throughput versus problem size for the best OpenMP and SYCL
# BK1 kernels at nq = 4, 6, 8: DoFs (nelmt * nm^3) log-spaced from DOF_MIN to
# DOF_MAX, one run of the harness per point, host-timed (best of NTESTS).
#
# usage:  scripts/study/dof_sweep.sh [results.csv]
# env:    DOF_MIN=1e3 DOF_MAX=1e7 PPD=4 (points per decade) NTESTS=20
#         OMP_KERNELS="4:gsI_b4 6:gsI_b1 8:gsI_b1"   nq:variant of bk1_offload_<nq> (ref: the reference kernel)
#         SYCL_KERNELS="4:inl_b4 6:inl_b1 8:inl_b1"  nq:variant of bk1_sycl_<nq>
# Needs build/a770/bk1_offload_<nq> and bk1_sycl_<nq> (scripts/build.sh all_bk1_offload all_bk1_sycl).
# CSV: api,nq,kernel,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev  (device columns: SYCL only)
# Also writes <results>.stream with the STREAM-like copy/triad bandwidths.

. "$(dirname "$0")/../common.sh"
OUT=${1:-dof_sweep.csv}
DOF_MIN=${DOF_MIN:-1e3}
DOF_MAX=${DOF_MAX:-1e7}
PPD=${PPD:-4}
NTESTS=${NTESTS:-20}
OMP_KERNELS=${OMP_KERNELS:-"4:gsI_b4 6:gsI_b1 8:gsI_b1"}
SYCL_KERNELS=${SYCL_KERNELS:-"4:inl_b4 6:inl_b1 8:inl_b1"}

nelmts() {   # nq -> the distinct element counts for the DoF targets
    python3 -c "
import math
lo, hi, ppd, nm3 = math.log10($DOF_MIN), math.log10($DOF_MAX), $PPD, ($1 - 1) ** 3
n = round((hi - lo) * ppd)
print(' '.join(sorted({str(max(1, round(10 ** (lo + i / ppd) / nm3))) for i in range(n + 1)}, key=int)))"
}

echo "api,nq,kernel,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev" > "$OUT"
for spec in $OMP_KERNELS; do
    nq=${spec%%:*}
    k=${spec#*:}
    for ne in $(nelmts "$nq"); do
        # the reference kernel runs only with BK_REF=1 (and then alone, as k)
        ref=0
        if [ "$k" = ref ]; then
            ref=1
        fi
        line=$(BK_REF=$ref BK_KERNEL=$k "$SCRIPTS/run.sh" "bk1_offload_$nq" "$ne" "$NTESTS" 2>&1 | grep -E "^$k +GDoF")
        g=$(echo "$line" | sed -n 's/.*GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ ]*\).*/\1,\2/p')
        echo "omp,$nq,$k,$ne,$(( ne * (nq - 1) ** 3 )),${g:-NaN,NaN},," | tee -a "$OUT"
    done
done
for spec in $SYCL_KERNELS; do
    nq=${spec%%:*}
    k=${spec#*:}
    for ne in $(nelmts "$nq"); do
        line=$(BK_KERNEL=$k "$SCRIPTS/run.sh" "bk1_sycl_$nq" "$ne" "$NTESTS" 2>&1 | grep -E "^$k +GDoF")
        g=$(echo "$line" | sed -n 's/.*GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ ]*\) *(device: GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ )]*\)).*/\1,\2,\3,\4/p')
        fail=$(echo "$line" | grep -c FAIL)
        [ "$fail" = 0 ] || g="NaN,NaN,NaN,NaN"
        echo "sycl,$nq,$k,$ne,$(( ne * (nq - 1) ** 3 )),${g:-NaN,NaN,NaN,NaN}" | tee -a "$OUT"
    done
done
"$SCRIPTS/run.sh" stream_omp > "$OUT.stream" 2>&1
grep -E "^(copy|triad|read|write|scale|add) " "$OUT.stream"
