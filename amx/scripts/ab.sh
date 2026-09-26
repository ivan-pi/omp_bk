#!/bin/bash
# ab.sh -- interleaved A/B of two builds of the AMX driver, median of R runs each
# usage:  A=./BK1_amx_before [B=<repo>/BK1_amx] [R=5] [ORDERS="4 8 14"] ./ab.sh
. "$(dirname "$0")/common.sh"
A=${A:?set A to the baseline binary}; B=${B:-$BK1_AMX}; R=${R:-5}
export BK_NOREF=1 BK_LAYOUT=${BK_LAYOUT:-soa}
median() { sort -g | awk '{a[NR]=$1} END{print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}'; }
for p in ${ORDERS:-4 8 14}; do
  n=$(nelmt_for_dofs 1e7 $p)
  ra=""; rb=""
  for ((i = 0; i < R; i++)); do
    ra="$ra $($A $p $n 3 | gdofs)"
    rb="$rb $($B $p $n 3 | gdofs)"
  done
  ma=$(echo $ra | tr ' ' '\n' | median); mb=$(echo $rb | tr ' ' '\n' | median)
  printf "p=%-2d  A median %.3f  B median %.3f  B/A = %.3f   (A:%s  B:%s)\n" $p $ma $mb $(awk -v a=$ma -v b=$mb 'BEGIN{print b/a}') "$ra" "$rb"
done
