#!/bin/bash
# bk5_sweep.sh -- BK5 throughput versus problem size: the given variant per
# API, order and tensor storage (G6 symmetric half, G9 full), DoFs (nelmt *
# nq^3) log-spaced from DOF_MIN to DOF_MAX, one run per point (OpenMP:
# build/a770/bk5_offload_<p>, SYCL: build/a770/bk5_sycl_<p>), best of NTESTS.
#
# usage:  KERNELS="omp:1:pt6f_b16 sycl:6:sf6c_b1x1 ..." scripts/study/bk5_sweep.sh [out.csv]
#         (<api>:<p>:<variant>, nq = p + 1; a spec without an API,
#         "1:pt6f_b16", is OpenMP)
# env:    DOF_MIN=1e3 DOF_MAX=1e8 PPD=4 NTESTS=20 MEM_MAX (common.sh; 1e8 DoFs at
#         p = 8 needs ~7.6 GB on the host: G6 + G9 + the double reference)
# CSV (the layout of dof_sweep.sh, plot_dof_sweep.py reads it):
#   api,nq,kernel,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev
#   api = <omp|sycl><6|9> (API and tensor factors); the *_dev columns are
#   SYCL's event-profiled device times (empty for OpenMP)
. "$(dirname "$0")/../common.sh"
OUT=${1:-bk5_sweep.csv}
DOF_MIN=${DOF_MIN:-1e3}
DOF_MAX=${DOF_MAX:-1e8}
PPD=${PPD:-4}
NTESTS=${NTESTS:-20}
echo "api,nq,kernel,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev" > "$OUT"
for spec in $KERNELS; do
    case $spec in
        *:*:*) lang=${spec%%:*}; rest=${spec#*:} ;;
        *) lang=omp; rest=$spec ;;
    esac
    p=${rest%%:*}
    nq=$((p + 1))
    k=${rest#*:}
    F=6
    case $k in sf9* | pt9* | ref9* | cu9* | tid9*) F=9 ;; esac
    bin=bk5_offload_$p
    if [ "$lang" = sycl ]; then
        bin=bk5_sycl_$p
    fi
    for ne in $(python3 -c "
import math
lo, hi, ppd, n3 = math.log10($DOF_MIN), math.log10($DOF_MAX), $PPD, $nq ** 3
n = round((hi - lo) * ppd)
print(' '.join(sorted({str(max(1, round(10 ** (lo + i / ppd) / n3))) for i in range(n + 1)}, key=int)))"); do
        line=$(BK_KERNEL=$k "$SCRIPTS/run.sh" "$bin" "$ne" "$NTESTS" 2>&1 | grep -E "^$k +GDoF")
        g=$(echo "$line" | sed -n 's/^[^ ]* *GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ ]*\).*/\1,\2/p')
        dev=$(echo "$line" | sed -n 's/.*device: GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ )]*\).*/\1,\2/p')
        if [ -z "$g" ] || echo "$line" | grep -q FAIL; then
            g="NaN,NaN"
            dev=""
        fi
        echo "$lang$F,$nq,$k,$ne,$(( ne * nq ** 3 )),$g,${dev:-,}" | tee -a "$OUT"
    done
done
