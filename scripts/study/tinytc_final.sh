#!/bin/bash
# tinytc_final.sh -- the tinytc kernels at one size with a chosen work-group
# size per BK, layout and nq, REPS runs each, checked against the reference.
#
# usage:  scripts/study/tinytc_final.sh <picks.txt> [out.csv]
#         picks.txt: lines "<bk> <layout> <p> <wgs> [<sg> <B>]" (e.g. the best of
#         tinytc_survey.sh; sg, B: BK3's sub-group and block sizes, default 16)
# env:    DOFS=56234132 (the DoF-sweep point near 5.6e7) REPS=3 NTESTS=20
# CSV:    bk,layout,p,nq,wgs,rep,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev,relerr,status,sg,B
. "$(dirname "$0")/../common.sh"
PICKS=$1
OUT=${2:-tinytc_final.csv}
DOFS=${DOFS:-56234132}
REPS=${REPS:-3}
NTESTS=${NTESTS:-20}
echo "bk,layout,p,nq,wgs,rep,nelmt,dofs,gdofs,gbs,gdofs_dev,gbs_dev,relerr,status,sg,B" > "$OUT"
while read -r bk layout p wgs sg B; do
    sg=${sg:-16}
    B=${B:-16}
    if [ "$bk" = 5 ]; then
        nq=$((p + 1)); per=$(( nq ** 3 ))
    else
        nq=$((p + 2)); per=$(( (nq - 1) ** 3 ))
    fi
    ne=$(python3 -c "print(max(32, round($DOFS / $per) // 32 * 32))")
    for rep in $(seq 1 "$REPS"); do
        line=$(TT_LAYOUT=$layout TT_WGS=$wgs TT_SG=$sg TT_B=$B "$SCRIPTS/run.sh" bk_tinytc "$bk" "$p" "$ne" "$NTESTS" 2>&1 | grep -E "^tt$bk")
        vals=$(echo "$line" | sed -n 's/.*GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ ]*\) *(device: GDoF\/s = *\([^ ]*\) *GB\/s = *\([^ )]*\)) *relerr = *\([^ ]*\).*/\1,\2,\3,\4,\5/p')
        st=ok
        if [ -z "$vals" ]; then
            vals="NaN,NaN,NaN,NaN,NaN"; st=error
        elif echo "$line" | grep -q FAIL; then
            st=FAIL
        fi
        echo "$bk,$layout,$p,$nq,$wgs,$rep,$ne,$((ne * per)),$vals,$st,$sg,$B" | tee -a "$OUT"
    done
done < "$PICKS"
