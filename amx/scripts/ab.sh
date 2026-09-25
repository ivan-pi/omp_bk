#!/bin/bash
# ab.sh -- interleaved A/B of two binaries, median of R runs each
A=${A:-./bk1_amx_v6}; B=${B:-./bk1_amx}; R=${R:-5}
export BK_NOREF=1 BK_LAYOUT=${BK_LAYOUT:-soa}
median() { sort -g | awk '{a[NR]=$1} END{print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}'; }
for p in ${ORDERS:-4 8 14}; do
  n=$(( 10000000 / ((p+1)*(p+1)*(p+1)) ))
  ra=""; rb=""
  for ((i = 0; i < R; i++)); do
    ra="$ra $($A $p $n 3 | sed -n 's/.*GDoF\/s = \([0-9.e-]*\).*/\1/p')"
    rb="$rb $($B $p $n 3 | sed -n 's/.*GDoF\/s = \([0-9.e-]*\).*/\1/p')"
  done
  ma=$(echo $ra | tr ' ' '\n' | median); mb=$(echo $rb | tr ' ' '\n' | median)
  printf "p=%-2d  A median %.3f  B median %.3f  B/A = %.3f   (A:%s  B:%s)\n" $p $ma $mb $(awk -v a=$ma -v b=$mb 'BEGIN{print b/a}') "$ra" "$rb"
done
