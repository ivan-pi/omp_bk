#!/bin/bash
# spec_compare.sh -- a SYCL kernel shaped by specialization constants (one
# JIT binary for every order) against its ahead-of-time twins (one binary per
# order): build time of each shape with cold and warm caches, and device
# GDoF/s of both.  The results of each kernel go in the table of
# sycl/README.md ("Specialization constants").
#
# usage:  scripts/study/spec_compare.sh <aot> <sc> [out.csv]
#   e.g.  scripts/study/spec_compare.sh bk1_sycl bk1_sycl_sc results/a770/spec_bk1_a770.csv
# env:    ORDERS="1 2 3 4 5 6 7 8"  DOFS=2e7 (elements: DOFS / (p + 1)^3)
#         NTESTS=20  KERNEL=<BK_KERNEL for <sc>, default all>
#
# The programs follow the convention of bk1_sycl_sc (sycl/README.md):
#   $BUILD_DIR/<aot>_<p> [nelmt] [ntests]   and   $BUILD_DIR/<sc> <p> [nelmt] [ntests]
# print "<name> GDoF/s = ...  (device: GDoF/s = x ...)  relerr = r" per
# variant, and <sc> prints after each "<name> shape <shape> aot <aot name|->
# build <s> s first <s> s".  Per order: <sc> with both the SYCL and the
# driver's compiler caches cold (NEO_CACHE_PERSISTENT=0, a new SYCL_CACHE_DIR;
# only its build times are used, so a small benchmark-only run), <sc> again
# with the SYCL cache warm (checked: its rates and relerr are the record),
# then <aot>_<p> with the twins only (benchmark only).
# CSV: kernel,p,name,shape,aot,sc_dev,aot_dev,ratio,build_cold,build_warm,relerr

. "$(dirname "$0")/../common.sh"
AOT=$1
SC=$2
OUT=${3:-spec_${SC}.csv}
if [ -z "$AOT" ] || [ -z "$SC" ]; then
    sed -n '2,/^# CSV/p' "$0" >&2
    exit 2
fi
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
DOFS=${DOFS:-2e7}
NTESTS=${NTESTS:-20}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "kernel,p,name,shape,aot,sc_dev,aot_dev,ratio,build_cold,build_warm,relerr" > "$OUT"
for p in $ORDERS; do
    ne=$(python3 -c "print(max(1, round($DOFS / ($p + 1) ** 3)))")
    if [ ! -x "$BUILD_DIR/$SC" ] || [ ! -x "$BUILD_DIR/${AOT}_$p" ]; then
        echo "skip p=$p: $SC or ${AOT}_$p not in $BUILD_DIR" >&2
        continue
    fi
    rm -rf "$TMP/cache"
    export NEO_CACHE_PERSISTENT=0 SYCL_CACHE_DIR="$TMP/cache" BK_KERNEL="${KERNEL:-}"
    BK_CHECK=0 "$SCRIPTS/run.sh" "$SC" "$p" 1237 1 > "$TMP/cold.txt" 2>&1
    "$SCRIPTS/run.sh" "$SC" "$p" "$ne" "$NTESTS" > "$TMP/warm.txt" 2>&1
    unset NEO_CACHE_PERSISTENT SYCL_CACHE_DIR BK_KERNEL
    twins=$(awk '$2 == "shape" && $5 != "-" {print $5}' "$TMP/warm.txt" | sort -u | tr '\n' ' ')
    BK_CHECK=0 BK_KERNEL="$twins" "$SCRIPTS/run.sh" "${AOT}_$p" "$ne" "$NTESTS" > "$TMP/aot.txt" 2>&1
    python3 - "$SC" "$p" "$TMP" >> "$OUT" <<'PY'
import re, sys
kernel, p, tmp = sys.argv[1], sys.argv[2], sys.argv[3]
res = re.compile(r'(\S+)\s+GDoF/s =.*device: GDoF/s =\s*([\d.]+).*relerr = (\S+?)(\s+FAIL)?$')
info = re.compile(r'(\S+)\s+shape (\S+) aot (\S+) build (\S+) s')
def read(name):
    rates, infos = {}, {}
    for line in open(f'{tmp}/{name}.txt'):
        m = res.match(line.rstrip())
        if m:
            rates[m[1]] = (float(m[2]), 'FAIL' if m[4] else m[3])
        m = info.match(line)
        if m:
            infos[m[1]] = (m[2], m[3], m[4])
    return rates, infos
_, cold = read('cold')
warm_rates, warm = read('warm')
aot_rates, _ = read('aot')
for name, (shape, twin, build) in warm.items():
    sc_dev, relerr = warm_rates.get(name, (0.0, 'missing'))
    aot_dev = aot_rates.get(twin, (0.0, ''))[0]
    ratio = f'{sc_dev / aot_dev:.3f}' if aot_dev > 0 else ''
    print(f'{kernel},{p},{name},{shape},{twin},{sc_dev},{aot_dev or ""},{ratio},'
          f'{cold.get(name, ("", "", ""))[2]},{build},{relerr}')
PY
    echo "p=$p done: $(grep '^JIT' "$TMP/cold.txt") cold, $(grep '^JIT' "$TMP/warm.txt" | sed 's/^JIT: //') warm" >&2
done
python3 - "$OUT" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(f"{'p':>2}  {'shapes':>6}  {'median sc/aot':>13}  {'min':>5}  {'max':>5}  {'cold build/shape':>16}  {'warm':>6}  fails")
for p in sorted({int(r['p']) for r in rows}):
    rs = [r for r in rows if int(r['p']) == p]
    ratios = [float(r['ratio']) for r in rs if r['ratio']]
    cold = [float(r['build_cold']) for r in rs if r['build_cold'] not in ('', '-')]
    warm = [float(r['build_warm']) for r in rs if r['build_warm'] not in ('', '-')]
    fails = sum(r['relerr'] in ('FAIL', 'missing') for r in rs)
    if ratios:
        print(f"{p:>2}  {len(rs):>6}  {statistics.median(ratios):>13.3f}  {min(ratios):>5.2f}  {max(ratios):>5.2f}  "
              f"{statistics.median(cold) if cold else 0:>14.3f} s  {statistics.median(warm) if warm else 0:>5.3f}s  {fails}")
PY
