#!/bin/bash
# tune.sh -- for each polynomial order, try layout x path x batch size (and
# optionally thread count), report the best configuration, and write all
# measurements to a CSV.  Works with bash 3.2 (macOS /bin/bash).
#
# usage:  ./tune.sh [dofs=1e7] [ntests=3]
#         Every configuration runs the same number of degrees of freedom:
#         nelmt = dofs / (p+1)^3, so the time per point is about constant
#         across orders instead of growing like p^3.  The default is above the
#         plateau onset seen in the throughput sweeps (~5e6 DoF).
# env:    ORDERS="1 2 ... 14"     polynomial orders (binary supports 1..14)
#         NELMT=<n>               fixed element count instead of dofs/(p+1)^3
#         BATCHES="16 32 64"     batch sizes for the sum-factorized element-major
#                                path; the dense path and the SoA layout always
#                                use 16 and are measured once
#         DENSE_MAX=4            try the dense path for p <= DENSE_MAX
#         THREADS=""             thread counts to try; empty = the binary's default
#         REPEATS=2              full repetitions of each configuration (best kept)
#         OUT=tune.csv           all measurements
#
# The layout is not really a tunable -- it is set by how the application stores
# its data -- so the summary reports the best configuration for each layout
# separately as well as the overall best.
. "$(dirname "$0")/common.sh"
DOFS=${1:-1e7}
T=${2:-3}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14"}
BATCHES=${BATCHES:-"16 32 64"}
DENSE_MAX=${DENSE_MAX:-4}
THREADS=${THREADS:-""}
REPEATS=${REPEATS:-2}
OUT=${OUT:-tune.csv}
export BK_NOREF=1

echo "p,threads,layout,dense,batch,gdofs,gbs" > "$OUT"

gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 > b + 0) }'; }   # a > b ?

measure() {   # measure <p> <threads> <layout> <dense> <batch> -> best "gdofs gbs" over REPEATS
  local p=$1 th=$2 layout=$3 dense=$4 batch=$5 best="" r i
  for ((i = 0; i < REPEATS; i++)); do
    r=$(env ${th:+OMP_NUM_THREADS=$th} BK_LAYOUT=$layout BK_DENSE=$dense BK_BATCH=$batch $BK1_AMX $p $N $T | rates)
    [ -z "$r" ] && continue
    if [ -z "$best" ] || gt "${r%% *}" "${best%% *}"; then best=$r; fi
  done
  echo "$best"
}

summary=""
for p in $ORDERS; do
  if [ -n "$NELMT" ]; then N=$NELMT
  else N=$(nelmt_for_dofs "$DOFS" "$p"); [ "$N" -lt 64 ] && N=64
  fi
  best_all=""; cfg_all=""
  best_aos=""; cfg_aos=""
  best_soa=""; cfg_soa=""
  for th in ${THREADS:-default}; do
    if [ "$th" = default ]; then thv=""; else thv=$th; fi
    for layout in aos soa; do
      denses="0"; [ "$p" -le "$DENSE_MAX" ] && denses="0 1"
      for dense in $denses; do
        # the batch is a tunable only for sumfact + element-major
        batches=$BATCHES
        if [ "$dense" = 1 ] || [ "$layout" = soa ]; then batches=16; fi
        for batch in $batches; do
          r=$(measure "$p" "$thv" $layout $dense $batch)
          if [ -z "$r" ]; then
            echo "$p,${thv:-default},$layout,$dense,$batch,NaN,NaN" | tee -a "$OUT"; continue
          fi
          set -- $r; g=$1; b=$2
          echo "$p,${thv:-default},$layout,$dense,$batch,$g,$b" | tee -a "$OUT"
          cfg="threads=${thv:-default} layout=$layout dense=$dense batch=$batch"
          if [ "$layout" = aos ]; then
            if [ -z "$best_aos" ] || gt "$g" "$best_aos"; then best_aos=$g; cfg_aos=$cfg; fi
          else
            if [ -z "$best_soa" ] || gt "$g" "$best_soa"; then best_soa=$g; cfg_soa=$cfg; fi
          fi
          if [ -z "$best_all" ] || gt "$g" "$best_all"; then best_all=$g; cfg_all=$cfg; fi
        done
      done
    done
  done
  summary="$summary
p=$p  (nelmt=$N)  best: $best_all GDoF/s  ($cfg_all)
       aos: $best_aos GDoF/s  ($cfg_aos)
       soa: $best_soa GDoF/s  ($cfg_soa)"
done

echo
echo "=== best configurations (${NELMT:+nelmt=$NELMT}${NELMT:-dofs=$DOFS}, ntests=$T, repeats=$REPEATS) ==="
echo "$summary" | sed 1d
echo "all measurements in $OUT"
