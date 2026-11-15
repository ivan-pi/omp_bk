#!/bin/bash
# variant_survey.sh -- every kernel variant of the OpenMP (bk1_offload) and SYCL
# (bk1_sycl) harnesses at a fixed problem size per order, REPS runs each, so
# the fastest variant per polynomial degree can be read off.
#
# usage:  scripts/study/variant_survey.sh [survey.csv]
# env:    NQS="3 4 5 6 7 8 9 10"  DOFS=1e8  REPS=3  NTESTS=10
# Needs build/a770/bk1_offload_<nq> and bk1_sycl_<nq>.
# CSV: api,nq,p,variant,rep,gdofs,gdofs_dev,check
#   check: OpenMP maxdiff against the reference kernel ("reference" for ref
#   itself); SYCL relative error against the double CPU reference, or FAIL.
# Prints the median ranking per order (top 5 per API) at the end.

. "$(dirname "$0")/../common.sh"
OUT=${1:-variant_survey.csv}
NQS=${NQS:-"3 4 5 6 7 8 9 10"}
DOFS=${DOFS:-1e8}
REPS=${REPS:-3}
NTESTS=${NTESTS:-10}

echo "api,nq,p,variant,rep,gdofs,gdofs_dev,check" > "$OUT"
for nq in $NQS; do
    ne=$(python3 -c "print(max(1, round($DOFS / ($nq - 1) ** 3)))")
    for rep in $(seq 1 "$REPS"); do
        "$SCRIPTS/run.sh" "bk1_offload_$nq" "$ne" "$NTESTS" 2>&1 | grep -E "GDoF/s" | \
            awk -v nq="$nq" -v rep="$rep" '{chk = ($NF == "(reference)") ? "reference" : $NF;
                 printf "omp,%d,%d,%s,%d,%s,,%s\n", nq, nq - 2, $1, rep, $4, chk}' >> "$OUT"
        "$SCRIPTS/run.sh" "bk1_sycl_$nq" "$ne" "$NTESTS" 2>&1 | grep -E "GDoF/s" | sed 's/(device: GDoF\/s =//' | \
            awk -v nq="$nq" -v rep="$rep" '{chk = ($NF == "FAIL") ? "FAIL" : $(NF);
                 printf "sycl,%d,%d,%s,%d,%s,%s,%s\n", nq, nq - 2, $1, rep, $4, $8, chk}' >> "$OUT"
    done
    echo "nq=$nq done"
done
python3 - "$OUT" <<'PY'
import csv, sys, statistics, collections
host = collections.defaultdict(list); dev = collections.defaultdict(list); bad = set()
for r in csv.DictReader(open(sys.argv[1])):
    k = (r["api"], int(r["p"]), r["variant"])
    host[k].append(float(r["gdofs"]))
    if r["gdofs_dev"]:
        dev[k].append(float(r["gdofs_dev"]))
    c = r["check"]
    if c == "FAIL" or (c not in ("reference",) and not c.startswith("0.00") and r["api"] == "omp"):
        bad.add(k)
for p in sorted({k[1] for k in host}):
    print(f"p = {p}")
    for api in ("omp", "sycl"):
        ks = sorted((k for k in host if k[0] == api and k[1] == p), key=lambda k: -statistics.median(host[k]))
        top = ", ".join("%s %.1f%s%s" % (k[2], statistics.median(host[k]),
                        " (dev %.1f)" % statistics.median(dev[k]) if k in dev else "", " BAD" if k in bad else "") for k in ks[:5])
        print(f"  {api:5s} {top}")
PY
