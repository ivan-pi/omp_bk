#!/usr/bin/env bash
# sweep.sh -- run the drivers over a range of orders into one CSV, then print
# a table.  Nothing is parsed: the drivers append their own rows when BK_CSV
# names a file (driver,kernel,nq,nelmt,batch,gdofs,gbs,norm,error,ok), and
# table.py pivots that file.
#
#   cuda/scripts/sweep.sh sweep.csv ./BK1_ff_cuda_plain [./BK1_ff ...]
#   ORDERS="2 3 4 5 6 7 8" REPEAT=2 NELMT=1000000 NTESTS=5 cuda/scripts/sweep.sh sweep.csv ./BK1_ff_cuda_plain
#   BK_RANDOM=1 cuda/scripts/sweep.sh random.csv ./BK1_ff_cuda_plain
#
# The first argument is the CSV (appended to if it exists), the rest are
# driver binaries, each run as  binary p NELMT NTESTS.  BATCH=n passes a
# fourth argument (elements per block) to the CUDA driver.  Exit status 1 if
# any run failed its check.
set -uo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 out.csv driver [driver...]" >&2
    exit 2
fi
csv=$1
shift
ORDERS=${ORDERS:-2 3 4 5 6 7 8}
REPEAT=${REPEAT:-1}
NELMT=${NELMT:-1000000}
NTESTS=${NTESTS:-5}
status=0
for p in $ORDERS; do
    for run in $(seq 1 "$REPEAT"); do
        for bin in "$@"; do
            extra=
            if [ -n "${BATCH:-}" ] && [[ "$bin" == *cuda* ]]; then
                extra=$BATCH
            fi
            echo "== $bin $p $NELMT $NTESTS $extra (run $run)"
            # shellcheck disable=SC2086
            if ! BK_CSV=$csv "$bin" "$p" "$NELMT" "$NTESTS" $extra; then
                status=1
            fi
        done
    done
done
echo
"$(dirname "$0")/table.py" "$csv"
if [ $status -ne 0 ]; then
    echo "FAIL: a run failed its check"
fi
exit $status
