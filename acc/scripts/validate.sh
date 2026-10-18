#!/bin/bash
# validate.sh -- check the OpenACC executables (acc/BK?_acc; every kernel of
# BK1_acc, selected with BK_KERNEL) over all supported orders.  Exit status
# is non-zero on any failure.
#
# usage: make BK1 BK3 BK5 && make -C acc all host && acc/scripts/validate.sh [nelmt=1234] [tol=1e-5]
# env:   BK1_KERNELS="elem gang1 gang2 gang3 worker kernels batch"   kernels of BK1_acc to check
#
# Two checks per executable and order:
#  (a) constant data: the norm must match the OpenMP executable in the
#      repository root (BK1, BK3, BK5; `make BK1 BK3 BK5`, or with nvc++
#      `make CXX=nvc++ CXXFLAGS="-O3 -std=c++17 -mp"`).  Where the
#      constant-data output cancels catastrophically (BK1 at p = 1: outputs
#      ~1e-5 from inputs of 3.0) the norm is reported but not enforced.
#      Skipped with a note when the OpenMP executable is absent.
#  (b) random data (BK_RANDOM=1): the norm of the offloaded build must match
#      the serial host build of the same source (acc/*_host, from
#      `make -C acc host`).  This is the check with discriminating power
#      for the OpenACC directives (privatisation, data movement, races);
#      it is skipped with a note when the host build is absent.  Each
#      kernel is compared with its own host build: the batch kernel of
#      BK1_acc reads JxW in a different point order, so its random-data norm
#      differs from the other kernels' by design.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACC=$ROOT/acc
N=${1:-1234}          # not a power of two, so no accidental alignment
TOL=${2:-1e-5}

pass=0; fail=0; skip=0; skip_ref=0

norm() { "$@" 1 | sed -n '2s/norm = //p'; }   # norm <exe> <order> <nelmt>  (ntests = 1)

# relerr <a> <b> -> |a - b| / |b|
relerr() { awk -v a="$1" -v b="$2" 'BEGIN { d = a - b; if (d < 0) { d = -d }; print d / (b != 0 ? (b < 0 ? -b : b) : 1) }'; }

# check <kernel> <acc-executable> <variant> <order> <ndof>
#   kernel: BK1, BK3 or BK5; variant: BK_KERNEL for the executable ("" = default)
check() {
  local K=$1 A=$2 V=$3 P=$4 ndof=$5
  local ref=$ROOT/$K acc=$ACC/$A host=$ACC/${A}_host
  local label=$A
  if [ -n "$V" ]; then
    label="$A:$V"
    export BK_KERNEL=$V
  else
    unset BK_KERNEL
  fi

  # (a) constant data against the OpenMP executable
  local n_ref n_acc verdict
  if [ -x "$ref" ]; then
    n_ref=$(norm $ref $P $N)
    n_acc=$(norm $acc $P $N)
    verdict=$(awk -v a="$n_acc" -v b="$n_ref" -v t="$TOL" -v ndof="$ndof" 'BEGIN {
      if (a == "" || b == "")        { print "FAIL"; exit }
      d = (a > b ? a - b : b - a) / (b > 0 ? b : 1);
      cancel = (b / sqrt(ndof)) / 3.0 < 1e-2;
      if (d <= t + 0)                print "ok";
      else if (cancel)               print "ok(norm cancels)";
      else                           print "FAIL"; }')
    case $verdict in
      ok*) printf "%-18s %-15s order=%d constant  norm=%-14s (%s: %s)\n" "$verdict" $label $P "$n_acc" $K "$n_ref"
           pass=$((pass+1));;
      *)   printf "%-18s %-15s order=%d constant  norm=%-14s (%s: %s)\n" "FAIL" $label $P "$n_acc" $K "$n_ref"
           fail=$((fail+1));;
    esac
  else
    skip_ref=$((skip_ref+1))
  fi

  # (b) random data against the serial host build of the same source
  if [ ! -x "$host" ]; then
    skip=$((skip+1))
    return
  fi
  local r_acc r_host rel
  r_acc=$(BK_RANDOM=1 norm $acc $P $N)
  r_host=$(BK_RANDOM=1 norm $host $P $N)
  rel=$(relerr "$r_acc" "$r_host")
  if [ -n "$r_acc" ] && [ -n "$r_host" ] && awk -v r="$rel" -v t="$TOL" 'BEGIN { exit !(r <= t) }'; then
    printf "%-18s %-15s order=%d random    norm=%-14s rel=%s\n" "ok" $label $P "$r_acc" "$rel"
    pass=$((pass+1))
  else
    printf "%-18s %-15s order=%d random    norm=%-14s (host: %s) rel=%s\n" "FAIL" $label $P "$r_acc" "$r_host" "$rel"
    fail=$((fail+1))
  fi
}

for exe in $ACC/BK1_acc $ACC/BK3_acc $ACC/BK5_acc; do
  if [ ! -x "$exe" ]; then
    echo "missing executable: $exe  (make -C acc)" >&2
    exit 2
  fi
done

BK1_KERNELS=${BK1_KERNELS:-"elem gang1 gang2 gang3 worker kernels batch"}

for P in 1 2 3 4 5 6 7 8; do          # BK1, BK3: p; nm = p + 1 modes per direction
  for V in $BK1_KERNELS; do
    check BK1 BK1_acc $V $P $(( N * (P+1)*(P+1)*(P+1) ))
  done
  check BK3 BK3_acc "" $P $(( N * (P+1)*(P+1)*(P+1) ))
done
for NQ in 2 3 4 5 6 7 8; do           # BK5: nq directly
  check BK5 BK5_acc "" $NQ $(( N * NQ*NQ*NQ ))
done

if [ $skip_ref -gt 0 ]; then
  echo "note: $skip_ref constant-data checks skipped (no BK1/BK3/BK5 in the repository root; run make BK1 BK3 BK5)"
fi
if [ $skip -gt 0 ]; then
  echo "note: $skip random-data checks skipped (no acc/*_host; run make -C acc host)"
fi
echo "$pass passed, $fail failed  (nelmt=$N, tol=$TOL)"
exit $((fail != 0))
