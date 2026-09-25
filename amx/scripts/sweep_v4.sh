for p in 1 2 3; do for d in 0 1; do for l in aos soa; do
  printf "p=%d dense=%d %s  " $p $d $l
  OMP_NUM_THREADS=6 BK_DENSE=$d BK_LAYOUT=$l ./bk1_amx $p 100000 3 | sed -n 's/.*GDoF\/s = \([0-9.e-]*\).*/\1/p'
done; done; done
for p in 4 6 8; do for l in aos soa; do
  printf "p=%d %s  " $p $l
  OMP_NUM_THREADS=6 BK_LAYOUT=$l ./bk1_amx $p 100000 3 | sed -n 's/.*GDoF\/s = \([0-9.e-]*\).*/\1/p'
done; done
