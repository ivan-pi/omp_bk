#!/bin/bash
# tinytc_survey.sh -- the tinytc BK1 and BK5 kernels (build/a770/bk_tinytc) over
# orders, layouts and work-group sizes at a fixed size, one run each, every
# result checked against the double-precision reference.
#
# usage:  scripts/study/tinytc_survey.sh [out.csv]
# env:    DOFS=1e7 NTESTS=10 LAYOUTS="ec blk" WGS="0 64 128 256 512 1024"
#         (0: the original driver's table)  ORDERS="1 ... 8" (p; BK1: nq = p + 2,
#         BK5: nq = p + 1)
# CSV:    bk,layout,p,nq,wgs,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev,relerr,status
#         status: ok, FAIL (wrong result) or error (did not run, e.g. too
#         much SLM for a 64 KB Xe-core)
. "$(dirname "$0")/../common.sh"
OUT=${1:-tinytc_survey.csv}
DOFS=${DOFS:-1e7}
NTESTS=${NTESTS:-10}
LAYOUTS=${LAYOUTS:-"ec blk"}
WGS=${WGS:-"0 64 128 256 512 1024"}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
echo "bk,layout,p,nq,wgs,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev,relerr,status" > "$OUT"
for bk in 1 5; do
    for p in $ORDERS; do
        if [ $bk = 1 ]; then
            nq=$((p + 2)); per=$(( (nq - 1) ** 3 ))
        else
            nq=$((p + 1)); per=$(( nq ** 3 ))
        fi
        ne=$(python3 -c "print(max(16, round($DOFS / $per) // 16 * 16))")
        for layout in $LAYOUTS; do
            for w in $WGS; do
                env=(TT_LAYOUT=$layout)
                if [ "$w" != 0 ]; then
                    env+=(TT_WGS=$w)
                fi
                log=$(env "${env[@]}" "$SCRIPTS/run.sh" bk_tinytc $bk $p $ne $NTESTS 2>&1)
                wg=$(echo "$log" | sed -n 's/.*wgs = \([0-9]*\).*/\1/p' | head -1)
                line=$(echo "$log" | grep -E "^tt$bk")
                if [ -z "$line" ]; then
                    echo "$bk,$layout,$p,$nq,${wg:-$w},$ne,$((ne * per)),NaN,NaN,NaN,NaN,NaN,error" | tee -a "$OUT"
                    continue
                fi
                vals=$(echo "$line" | sed -n 's/.*GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ ]*\) *(device: GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ )]*\)) *relerr = *\([^ ]*\).*/\1,\2,\3,\4,\5/p')
                st=ok
                if echo "$line" | grep -q FAIL; then
                    st=FAIL
                fi
                echo "$bk,$layout,$p,$nq,$wg,$ne,$((ne * per)),$vals,$st" | tee -a "$OUT"
            done
        done
    done
done
