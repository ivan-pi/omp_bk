#!/bin/bash
# validate.sh -- check the OpenACC executables (acc/BK?_acc) over all supported
# orders.  Exit status is non-zero on any failure.
#
# usage: make BK1 BK3 BK5 && make -C acc all host && acc/scripts/validate.sh [nelmt=1234] [tol=1e-5]
#
# Two checks per order:
#  (a) constant data: the norm must match the OpenMP executable in the
#      repository root (BK1, BK3, BK5).  Where the constant-data output
#      cancels catastrophically (BK1 at p = 1: outputs ~1e-5 from inputs
#      of 3.0) the norm is reported but not enforced.
#  (b) random data (BK_RANDOM=1): the norm of the offloaded build must match
#      the serial host build of the same source (acc/BK?_acc_host, from
#      `make -C acc host`).  This is the check with discriminating power
#      for the OpenACC directives (privatisation, data movement, races);
#      it is skipped with a note when the host build is absent.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACC=$ROOT/acc
N=${1:-1234}          # not a power of two, so no accidental alignment
TOL=${2:-1e-5}

pass=0; fail=0; skip=0

norm() { "$@" 1 | sed -n '2s/norm = //p'; }   # norm <exe> <order> <nelmt>  (ntests = 1)

# relerr <a> <b> -> |a - b| / |b|
relerr() { awk -v a="$1" -v b="$2" 'BEGIN { d = a - b; if (d < 0) { d = -d }; print d / (b != 0 ? (b < 0 ? -b : b) : 1) }'; }

# check <kernel> <order> <ndof>   (kernel: BK1, BK3 or BK5)
check() {
  local K=$1 P=$2 ndof=$3
  local ref=$ROOT/$K acc=$ACC/${K}_acc host=$ACC/${K}_acc_host

  # (a) constant data against the OpenMP executable
  local n_ref n_acc verdict
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
    ok*) printf "%-18s %s order=%d constant  norm=%-14s (%s: %s)\n" "$verdict" $K $P "$n_acc" $K "$n_ref"
         pass=$((pass+1));;
    *)   printf "%-18s %s order=%d constant  norm=%-14s (%s: %s)\n" "FAIL" $K $P "$n_acc" $K "$n_ref"
         fail=$((fail+1));;
  esac

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
    printf "%-18s %s order=%d random    norm=%-14s rel=%s\n" "ok" $K $P "$r_acc" "$rel"
    pass=$((pass+1))
  else
    printf "%-18s %s order=%d random    norm=%-14s (host: %s) rel=%s\n" "FAIL" $K $P "$r_acc" "$r_host" "$rel"
    fail=$((fail+1))
  fi
}

for K in BK1 BK3 BK5; do
  for exe in $ROOT/$K $ACC/${K}_acc; do
    if [ ! -x "$exe" ]; then
      echo "missing executable: $exe  (make $K; make -C acc)" >&2
      exit 2
    fi
  done
done

for P in 1 2 3 4 5 6 7 8; do          # BK1, BK3: p; nm = p + 1 modes per direction
  check BK1 $P $(( N * (P+1)*(P+1)*(P+1) ))
  check BK3 $P $(( N * (P+1)*(P+1)*(P+1) ))
done
for NQ in 2 3 4 5 6 7 8; do           # BK5: nq directly
  check BK5 $NQ $(( N * NQ*NQ*NQ ))
done

if [ $skip -gt 0 ]; then
  echo "note: $skip random-data checks skipped (no acc/BK?_acc_host; run make -C acc host)"
fi
echo "$pass passed, $fail failed  (nelmt=$N, tol=$TOL)"
exit $((fail != 0))
