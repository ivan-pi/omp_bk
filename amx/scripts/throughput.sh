#!/bin/bash
# throughput.sh -- CEED-style throughput sweep: GDoF/s versus problem size,
# log-spaced from DOF_MIN to DOF_MAX, for each polynomial order and kernel.
#
# usage:  ./throughput.sh [results.csv]        (high_order.sh: the same for p = 9..14)
# env:    ORDERS="1 2 3 4 5 6 7 8"   polynomial orders
#         DOF_MIN=1e4 DOF_MAX=1e8    problem-size range in degrees of freedom
#         PPD=4                      points per decade
#         NTESTS=5                   repetitions per point (driver takes the minimum)
#         KERNELS="serial omp omp_v neon_aos neon_soa amx_aos amx_soa"
#           serial = the reference kernel on one thread (BK_KERNEL=ref
#                    BK_PARALLEL=0 in the AMX binary)
#           omp    = BK1 from the Makefile (OpenMP target, host fallback; p <= 8)
#           omp_v  = the reference loops interchanged for unit-stride inner
#                    loops (BK_KERNEL=refv in the AMX binary, OpenMP over elements)
#         SERIAL_MAX=1e7             cap for the serial kernel (it is slow)
#         OMP_NUM_THREADS            threads for omp / amx (default: cores - 1,
#                                    which is also the AMX driver's own default)
#
# Output CSV columns: kernel,p,target,nelmt,dofs,gdofs,gbs
#   target = the log-spaced size point, dofs = nelmt * (p+1)^3 actually run
# Each point is one run of the driver; parse failures are recorded as NaN.

. "$(dirname "$0")/common.sh"
OUT=${1:-results.csv}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
DOF_MIN=${DOF_MIN:-1e4}
DOF_MAX=${DOF_MAX:-1e8}
PPD=${PPD:-4}
NTESTS=${NTESTS:-5}
KERNELS=${KERNELS:-"serial omp omp_v neon_aos neon_soa amx_aos amx_soa"}
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
    serial)  out=$(BK_KERNEL=ref BK_PARALLEL=0 $BK1_AMX $p $n $NTESTS) ;;
    omp)     out=$($BK1     $p $n $NTESTS) ;;
    omp_v)    out=$(BK_KERNEL=refv $BK1_AMX $p $n $NTESTS) ;;
    neon_aos) out=$(BK_KERNEL=neon BK_LAYOUT=aos $BK1_AMX $p $n $NTESTS) ;;
    neon_soa) out=$(BK_KERNEL=neon BK_LAYOUT=soa $BK1_AMX $p $n $NTESTS) ;;
    amx_aos) out=$(BK_LAYOUT=aos $BK1_AMX $p $n $NTESTS) ;;
    amx_soa) out=$(BK_LAYOUT=soa $BK1_AMX $p $n $NTESTS) ;;
    *) echo "unknown kernel $k" >&2; return 1 ;;
  esac
  echo "$out" | rates
}

echo "kernel,p,target,nelmt,dofs,gdofs,gbs" > "$OUT"
for k in $KERNELS; do
  for p in $ORDERS; do
    nm3=$(( (p+1)*(p+1)*(p+1) ))
    for d in $sizes; do
      if [ "$k" = serial ] && awk -v d="$d" -v m="$SERIAL_MAX" 'BEGIN{exit !(d > m)}'; then continue; fi
      if [ "$k" = omp ] && [ "$p" -gt 8 ]; then continue; fi        # the BK1 executable stops at p = 8
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
