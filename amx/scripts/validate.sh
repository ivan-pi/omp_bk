#!/bin/bash
# validate.sh -- run bk1_amx over orders / paths / layouts and check every run
# against (a) its own serial-reference comparison on random data and (b) the
# norm printed by the plain bk1 executable on constant data.  Exit status is
# non-zero on any failure.
#
# usage: ./validate.sh [nelmt=1234] [tol=1e-5]
#
# The random-data check is the one with discriminating power (an index bug
# fails it; constant data cannot detect transpositions).  The constant-data
# norm cross-check ties the two executables together, but at p = 1 the test
# basis cancels catastrophically (outputs ~1e-5 from inputs of 3.0, ~3 fp32
# digits left), so there the norm is reported but not enforced.
#
# Threads: whatever OMP_NUM_THREADS says, else the binary's default (cores - 1).
# Batch size: the binary's default of 16, plus one explicit batch=32 line.
N=${1:-1234}          # not a multiple of 16, so partial lane chunks are exercised
TOL=${2:-1e-5}

pass=0; fail=0

check() {   # check <label> <env assignments...>
  local label=$1; shift

  # 1) random data: relative error against the serial reference inside the driver
  local rel
  rel=$(env "$@" BK_RANDOM=1 ./bk1_amx $P $N 1 | sed -n 's/.*relative = \([0-9.e+-]*\).*/\1/p')

  # 2) constant data: norm must match the standalone bk1 executable
  local n_amx n_ref
  n_amx=$(env "$@" ./bk1_amx $P $N 1 | sed -n '2s/norm = //p')
  n_ref=$(./bk1 $P $N 1 | sed -n '2s/norm = //p')

  # per-DoF output magnitude relative to the constant input (3.0); below 1e-2
  # the constant-data test cancels catastrophically -> norm check informational
  local ndof=$(( N * (P+1)*(P+1)*(P+1) ))
  local verdict
  verdict=$(awk -v r="$rel" -v t="$TOL" -v a="$n_amx" -v b="$n_ref" -v ndof="$ndof" 'BEGIN {
      d = (a > b ? a - b : b - a) / (b > 0 ? b : 1);
      cancel = (b / sqrt(ndof)) / 3.0 < 1e-2;
      if (r == "" || r + 0 > t + 0)   print "FAIL";
      else if (d <= t + 0)           print "ok";
      else if (cancel)               print "ok(norm cancels)";
      else                           print "FAIL"; }')

  case $verdict in
    ok*) printf "%-18s p=%d %-26s rel=%-11s norm=%s\n" "$verdict" $P "$label" "$rel" "$n_amx"
         pass=$((pass+1));;
    *)   printf "%-18s p=%d %-26s rel=%-11s norm=%s (bk1: %s)\n" "FAIL" $P "$label" "$rel" "$n_amx" "$n_ref"
         fail=$((fail+1));;
  esac
}

for P in 1 2 3 5 8; do
  for layout in aos soa; do
    check "sumfact $layout"        BK_DENSE=0 BK_LAYOUT=$layout
    if [ $P -le 3 ]; then
      check "dense $layout"        BK_DENSE=1 BK_LAYOUT=$layout
    fi
  done
  check "sumfact aos batch=32"     BK_DENSE=0 BK_LAYOUT=aos BK_BATCH=32
done

echo "$pass passed, $fail failed  (nelmt=$N, tol=$TOL, threads=${OMP_NUM_THREADS:-default})"
exit $((fail != 0))
