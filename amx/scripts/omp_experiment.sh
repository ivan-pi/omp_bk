#!/bin/bash
# usage: OMP_NUM_THREADS=6 ./run_experiment.sh [nelmt] [ntests]
N=${1:-100000}; T=${2:-3}
extract() { sed -n 's/.*GDoF\/s = \([0-9.e-]*\) GB\/s = \([0-9.e-]*\).*/\1 \2/p'; }

echo "threads = ${OMP_NUM_THREADS:-default}, nelmt = $N, ntests = $T"
{ echo "p kernel E GDoF/s GB/s"
  for p in 1 2 3 4 5 6 7 8; do
    echo "$p serial - $(./bk1     $p $N $T | extract)"
    echo "$p omp    - $(./bk1_omp $p $N $T | extract)"
    for E in 1 2 4 8 16 32; do
      echo "$p amx $E $(BK_BATCH=$E ./bk1_amx $p $N $T | extract)"
    done
  done
} | column -t

