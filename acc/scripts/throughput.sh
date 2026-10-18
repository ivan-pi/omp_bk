#!/bin/bash
# throughput.sh -- CEED-style throughput sweep of the OpenACC BK1 kernels:
# GDoF/s versus problem size, log-spaced from DOF_MIN to DOF_MAX, for each
# polynomial order and kernel.  One run of the driver per point.
#
# usage:  make -C acc && acc/scripts/throughput.sh [results.csv]
# env:    KERNELS="elem gang1 gang2 gang3 worker batch"
#             kernels of BK1_acc to sweep (BK_KERNEL names).  A token
#             name@path runs that kernel from another build of the driver,
#             e.g. batch@acc/BK1_acc_epg3 after
#               make -C acc BK1_acc -B DEFS=-DBK_EPG=3 && cp acc/BK1_acc acc/BK1_acc_epg3
#             and is recorded as batch@BK1_acc_epg3.
#         EXE=acc/BK1_acc            the driver for tokens without a path
#         ORDERS="1 2 3 4 5 6 7 8"   polynomial orders
#         DOF_MIN=1e4 DOF_MAX=1e8    problem-size range in degrees of freedom
#         PPD=4                      points per decade
#         NTESTS=10                  repetitions per point (driver takes the minimum)
#         BK_RANDOM                  passed through to the drivers (default: constant data)
#
# Output CSV columns: kernel,p,target,nelmt,dofs,gdofs,gbs
#   kernel = the KERNELS token (path reduced to its basename), target = the
#   log-spaced size point, dofs = nelmt * (p+1)^3 actually run.  Parse
#   failures are recorded as NaN.  The driver's stderr (its configuration
#   line) goes to the terminal once per kernel and order, not into the CSV.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACC=$ROOT/acc
OUT=${1:-results.csv}
KERNELS=${KERNELS:-"elem gang1 gang2 gang3 worker batch"}
EXE=${EXE:-$ACC/BK1_acc}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
DOF_MIN=${DOF_MIN:-1e4}
DOF_MAX=${DOF_MAX:-1e8}
PPD=${PPD:-4}
NTESTS=${NTESTS:-10}

# Parse the driver's "... GDoF/s = <g> GB/s = <b>" line from stdin -> "<g> <b>"
rates() { sed -n 's/.*GDoF\/s = \([0-9.e+-]*\) GB\/s = \([0-9.e+-]*\).*/\1 \2/p'; }

exe_of()  { case $1 in *@*) echo "${1#*@}" ;; *) echo "$EXE" ;; esac; }   # token -> driver path
name_of() { echo "${1%%@*}"; }                                              # token -> BK_KERNEL
label_of() { case $1 in *@*) echo "$(name_of $1)@$(basename "${1#*@}")" ;; *) echo "$1" ;; esac; }

for k in $KERNELS; do
  if [ ! -x "$(exe_of $k)" ]; then
    echo "missing executable: $(exe_of $k)  (make -C acc)" >&2
    exit 2
  fi
done

# log-spaced DoF targets
sizes=$(awk -v lo="$DOF_MIN" -v hi="$DOF_MAX" -v ppd="$PPD" 'BEGIN {
  n = int((log(hi) - log(lo)) / log(10) * ppd + 0.5);
  for (i = 0; i <= n; i++) printf "%d ", exp(log(lo) + i * log(10) / ppd) + 0.5 }')

echo "kernel,p,target,nelmt,dofs,gdofs,gbs" > "$OUT"
for k in $KERNELS; do
  exe=$(exe_of $k)
  export BK_KERNEL=$(name_of $k)
  name=$(label_of $k)
  for p in $ORDERS; do
    nm3=$(( (p+1)*(p+1)*(p+1) ))
    first=1
    for d in $sizes; do
      n=$(( (d + nm3/2) / nm3 ))
      if [ $n -lt 1 ]; then
        n=1
      fi
      dofs=$(( n * nm3 ))
      if [ $first -eq 1 ]; then
        r=$("$exe" $p $n $NTESTS | rates)         # stderr shown once: the configuration
        first=0
      else
        r=$("$exe" $p $n $NTESTS 2>/dev/null | rates)
      fi
      g=${r%% *}
      b=${r##* }
      if [ -z "$r" ]; then
        g=NaN
        b=NaN
      fi
      echo "$name,$p,$d,$n,$dofs,$g,$b" | tee -a "$OUT"
    done
  done
done
echo "wrote $OUT"
