#!/bin/bash
# bk1_pick.sh -- the fastest BK1 variants: every kernel variant of the OpenMP
# (bk1_offload) and SYCL (bk1_sycl) harnesses at a fixed problem size per
# order, REPS runs each, ranked per order by median host GDoF/s.
#
# usage:  scripts/study/bk1_pick.sh [out.csv]
# env:    ORDERS="1 2 3 4 5 6 7 8" (p, nq = p + 2)  DOFS=1e8  REPS=3  NTESTS=10
# Runs $BUILD_DIR/bk1_offload_<p> and bk1_sycl_<p> (BUILD_DIR, common.sh: e.g.
# build/gpu-sycl); programs missing from the tree are skipped.
# CSV: api,nq,p,variant,rep,gdofs,gdofs_dev,check
#   check: OpenMP maxdiff against the reference kernel ("reference" for ref
#   itself); SYCL relative error against the double CPU reference, or FAIL.
# Prints the median ranking per order (top 5 per API) at the end.

. "$(dirname "$0")/../common.sh"
OUT=${1:-bk1_pick.csv}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
DOFS=${DOFS:-1e8}
REPS=${REPS:-3}
NTESTS=${NTESTS:-10}

echo "api,nq,p,variant,rep,gdofs,gdofs_dev,check" > "$OUT"
for p in $ORDERS; do
    nq=$((p + 2))
    ne=$(python3 -c "print(max(1, round($DOFS / ($nq - 1) ** 3)))")
    for bin in bk1_offload_$p bk1_sycl_$p; do
        if [ ! -x "$BUILD_DIR/$bin" ]; then
            echo "skip $bin: not in $BUILD_DIR" >&2
        fi
    done
    for rep in $(seq 1 "$REPS"); do
        if [ -x "$BUILD_DIR/bk1_offload_$p" ]; then
            "$SCRIPTS/run.sh" "bk1_offload_$p" "$ne" "$NTESTS" 2>&1 | grep -E "GDoF/s" | \
                awk -v nq="$nq" -v rep="$rep" '{chk = ($NF == "(reference)") ? "reference" : $NF;
                     printf "omp,%d,%d,%s,%d,%s,,%s\n", nq, nq - 2, $1, rep, $4, chk}' >> "$OUT"
        fi
        if [ -x "$BUILD_DIR/bk1_sycl_$p" ]; then
            "$SCRIPTS/run.sh" "bk1_sycl_$p" "$ne" "$NTESTS" 2>&1 | grep -E "GDoF/s" | sed 's/(device: GDoF\/s =//' | \
                awk -v nq="$nq" -v rep="$rep" '{chk = ($NF == "FAIL") ? "FAIL" : $(NF);
                     printf "sycl,%d,%d,%s,%d,%s,%s,%s\n", nq, nq - 2, $1, rep, $4, $8, chk}' >> "$OUT"
        fi
    done
    echo "p=$p done"
done
python3 - "$OUT" <<'PY'
import csv, sys, statistics, collections
host = collections.defaultdict(list); dev = collections.defaultdict(list); bad = set()
for r in csv.DictReader(open(sys.argv[1])):
    k = (r["api"], int(r["p"]), r["variant"])
    try:
        g = float(r["gdofs"])
        gd = float(r["gdofs_dev"]) if r["gdofs_dev"] else None
    except ValueError:
        # a run that aborted in the middle of its line (e.g. a lost device)
        print(f"ignored malformed row: {','.join(r.values())}", file=sys.stderr)
        continue
    host[k].append(g)
    if gd is not None:
        dev[k].append(gd)
    c = r["check"]
    if c == "FAIL" or (c not in ("reference",) and not c.startswith("0.00") and r["api"] == "omp"):
        bad.add(k)
for p in sorted({k[1] for k in host}):
    print(f"p = {p}")
    for api in ("omp", "sycl"):
        ks = sorted((k for k in host if k[0] == api and k[1] == p), key=lambda k: -statistics.median(host[k]))
        if not ks:
            continue
        top = ", ".join("%s %.1f%s%s" % (k[2], statistics.median(host[k]),
                        " (dev %.1f)" % statistics.median(dev[k]) if k in dev else "", " BAD" if k in bad else "") for k in ks[:5])
        print(f"  {api:5s} {top}")
PY
