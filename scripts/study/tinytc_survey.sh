#!/bin/bash
# tinytc_survey.sh -- the tinytc BK1, BK3 and BK5 kernels (build/a770/bk_tinytc)
# over orders, layouts and work-group sizes (BK3: and sub-group and block
# sizes) at a fixed size, one run each, every result checked against the
# double-precision reference.
#
# usage:  scripts/study/tinytc_survey.sh [out.csv]
# env:    DOFS=1e7 NTESTS=10 LAYOUTS="ec blk" BK3_LAYOUTS="$LAYOUTS blk_ch"
#         (blk_ch: bk3_blk_ch.ir, the r direction chained in registers)
#         WGS="0 64 128 256 512 1024"
#         (0: the original driver's table)  ORDERS="1 ... 8" (p; BK1, BK3:
#         nq = p + 2, BK5: nq = p + 1)  BKS="1 3 5"
#         BK3_SB="8:8 8:16 16:16 16:32" (sub-group size:elements per block)
# CSV:    bk,layout,p,nq,wgs,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev,relerr,status,sg,B
#         status: ok, FAIL (wrong result) or error (did not run, e.g. too
#         much SLM for a 64 KB Xe-core)
. "$(dirname "$0")/../common.sh"
OUT=${1:-tinytc_survey.csv}
DOFS=${DOFS:-1e7}
NTESTS=${NTESTS:-10}
LAYOUTS=${LAYOUTS:-"ec blk"}
BK3_LAYOUTS=${BK3_LAYOUTS:-"$LAYOUTS blk_ch"}
WGS=${WGS:-"0 64 128 256 512 1024"}
ORDERS=${ORDERS:-"1 2 3 4 5 6 7 8"}
BKS=${BKS:-"1 3 5"}
BK3_SB=${BK3_SB:-"8:8 8:16 16:16 16:32"}
echo "bk,layout,p,nq,wgs,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev,relerr,status,sg,B" > "$OUT"
for bk in $BKS; do
    for p in $ORDERS; do
        if [ $bk = 5 ]; then
            nq=$((p + 1)); per=$(( nq ** 3 ))
        else
            nq=$((p + 2)); per=$(( (nq - 1) ** 3 ))
        fi
        ne=$(python3 -c "print(max(32, round($DOFS / $per) // 32 * 32))")
        sbs="16:16"
        layouts=$LAYOUTS
        if [ $bk = 3 ]; then
            sbs=$BK3_SB
            layouts=$BK3_LAYOUTS
        fi
        for layout in $layouts; do
            for sb in $sbs; do
                sg=${sb%:*}
                B=${sb#*:}
                for w in $WGS; do
                    if [ "$w" != 0 ] && [ $((w % sg)) != 0 ]; then
                        continue
                    fi
                    env=(TT_LAYOUT=$layout)
                    if [ $bk = 3 ]; then
                        env+=(TT_SG=$sg TT_B=$B)
                    fi
                    if [ "$w" != 0 ]; then
                        env+=(TT_WGS=$w)
                    fi
                    log=$(env "${env[@]}" "$SCRIPTS/run.sh" bk_tinytc $bk $p $ne $NTESTS 2>&1)
                    wg=$(echo "$log" | sed -n 's/.*wgs = \([0-9]*\).*/\1/p' | head -1)
                    line=$(echo "$log" | grep -E "^tt$bk")
                    if [ -z "$line" ]; then
                        echo "$bk,$layout,$p,$nq,${wg:-$w},$ne,$((ne * per)),NaN,NaN,NaN,NaN,NaN,error,$sg,$B" | tee -a "$OUT"
                        continue
                    fi
                    vals=$(echo "$line" | sed -n 's/.*GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ ]*\) *(device: GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ )]*\)) *relerr = *\([^ ]*\).*/\1,\2,\3,\4,\5/p')
                    st=ok
                    if echo "$line" | grep -q FAIL; then
                        st=FAIL
                    fi
                    echo "$bk,$layout,$p,$nq,$wg,$ne,$((ne * per)),$vals,$st,$sg,$B" | tee -a "$OUT"
                done
            done
        done
    done
done
