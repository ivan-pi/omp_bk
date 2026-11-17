#!/bin/bash
# check_gpu.sh -- correctness of every GPU kernel variant of the BK1, BK3 and
# BK5 harnesses, separate from the benchmarks (which run with BK_CHECK=0):
# for each order, bk<n>_offload_<p> and bk<n>_sycl_<p> (and bk1_sycl_sc) at a
# small odd element count (whole and partial batches), one launch per
# variant, checked against the double-precision references.
#
# usage:  scripts/check_gpu.sh
# env:    BKS="1 3 5"  ORDERS="1 2 3 4 5 6 7 8" (p)  NELMT=1237
# A variant fails on its "FAIL" mark (the harnesses compare against the
# tolerance of bk::fail_mark), an invalid result, or a run that exits with an
# error; variants beyond a device limit count as skipped.  Programs missing
# from $BUILD_DIR are skipped (bk_tinytc too: its shapes come from TT_*
# settings).  Prints one line per program and exits 1 if anything failed.

. "$(dirname "$0")/common.sh"
BKS=${BKS:-"1 3 5"}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
NELMT=${NELMT:-1237}
export BK_CHECK=1
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

failed=0
check() {   # <label> <program> <args...>
    local label=$1
    shift
    if [ ! -x "$BUILD_DIR/$1" ]; then
        echo "skip  $label: $1 not in $BUILD_DIR"
        return
    fi
    "$SCRIPTS/run.sh" "$@" > "$TMP" 2>&1
    python3 - "$TMP" "$?" "$label" <<'PY' || failed=1
import sys
path, rc, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n, bad, skipped = 0, [], 0
for line in open(path):
    name = line.split()[0] if line.split() else ""
    if "GDoF/s" in line:
        n += 1
        if line.rstrip().endswith("FAIL"):
            bad.append(name)
    elif "result invalid" in line:
        bad.append(name + " (invalid)")
    elif " skipped: " in line:
        skipped += 1   # beyond a device limit (work-group size, local memory)
ok = not bad and rc == 0 and n > 0
notes = ([f"{skipped} skipped: device limits"] if skipped else []) + bad + ([f"exit {rc}"] if rc else [])
print(f"{'ok' if ok else 'FAIL'}  {label}: {n} variants {' '.join(notes)}")
sys.exit(0 if ok else 1)
PY
}

for bk in $BKS; do
    for p in $ORDERS; do
        check "bk${bk}_offload_$p" "bk${bk}_offload_$p" "$NELMT" 1
        check "bk${bk}_sycl_$p" "bk${bk}_sycl_$p" "$NELMT" 1
        if [ "$bk" = 1 ]; then
            check "bk1_sycl_sc $p" bk1_sycl_sc "$p" "$NELMT" 1
        fi
    done
done
exit $failed
