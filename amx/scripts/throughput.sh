#!/bin/bash
# throughput.sh -- CEED-style throughput sweep: GDoF/s versus problem size,
# log-spaced from DOF_MIN to DOF_MAX, for each polynomial order and kernel.
#
# usage:  ./throughput.sh [results.csv]
# env:    ORDERS="1 2 3 4 5 6 7 8"   polynomial orders
#         DOF_MIN=1e4 DOF_MAX=1e8    problem-size range in degrees of freedom
#         PPD=4                      points per decade
#         NTESTS=5                   repetitions per point (driver takes the minimum)
#         KERNELS="serial omp amx_aos amx_soa"
#         SERIAL_MAX=1e7             cap for the serial kernel (it is slow)
#         OMP_NUM_THREADS            threads for omp / amx (default: cores - 1)
#
# Output CSV columns: kernel,p,target,nelmt,dofs,gdofs,gbs
#   target = the log-spaced size point, dofs = nelmt * (p+1)^3 actually run
# Each point is one run of the driver; parse failures are recorded as NaN.

OUT=${1:-results.csv}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
DOF_MIN=${DOF_MIN:-1e4}
DOF_MAX=${DOF_MAX:-1e8}
PPD=${PPD:-4}
NTESTS=${NTESTS:-5}
KERNELS=${KERNELS:-"serial omp amx_aos amx_soa"}
SERIAL_MAX=${SERIAL_MAX:-1e7}
ncpu=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( ncpu > 1 ? ncpu - 1 : 1 ))}
export BK_NOREF=1

# log-spaced DoF targets
sizes=$(awk -v lo="$DOF_MIN" -v hi="$DOF_MAX" -v ppd="$PPD" 'BEGIN {
  n = int((log(hi) - log(lo)) / log(10) * ppd + 0.5);
  for (i = 0; i <= n; i++) printf "%d ", exp(log(lo) + i * log(10) / ppd) + 0.5 }')

run() {   # run <kernel> <p> <nelmt>  -> "gdofs gbs"
  local k=$1 p=$2 n=$3 out
  case $k in
    serial)  out=$(./bk1     $p $n $NTESTS) ;;
    omp)     out=$(./bk1_omp $p $n $NTESTS) ;;
    amx_aos) out=$(BK_LAYOUT=aos ./bk1_amx $p $n $NTESTS) ;;
    amx_soa) out=$(BK_LAYOUT=soa ./bk1_amx $p $n $NTESTS) ;;
    *) echo "unknown kernel $k" >&2; return 1 ;;
  esac
  echo "$out" | sed -n 's/.*GDoF\/s = \([0-9.e+-]*\) GB\/s = \([0-9.e+-]*\).*/\1 \2/p'
}

echo "kernel,p,target,nelmt,dofs,gdofs,gbs" > "$OUT"
for k in $KERNELS; do
  for p in $ORDERS; do
    nm3=$(( (p+1)*(p+1)*(p+1) ))
    for d in $sizes; do
      if [ "$k" = serial ] && awk -v d="$d" -v m="$SERIAL_MAX" 'BEGIN{exit !(d > m)}'; then continue; fi
      n=$(( (d + nm3/2) / nm3 )); [ $n -lt 1 ] && n=1
      dofs=$(( n * nm3 ))
      r=$(run $k $p $n)
      g=${r%% *}; b=${r##* }
      [ -z "$r" ] && { g=NaN; b=NaN; }
      echo "$k,$p,$d,$n,$dofs,$g,$b" | tee -a "$OUT"
    done
  done
done
echo "wrote $OUT"
