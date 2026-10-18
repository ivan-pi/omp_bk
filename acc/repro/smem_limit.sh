#!/bin/bash
# smem_limit.sh -- compile smem_limit.cpp over a grid of array counts and sizes
# and report whether nvc++ placed the gang-private arrays in shared memory.
# usage: ./smem_limit.sh            env CXX (default nvc++), ACC (default -acc=gpu)
cd "$(dirname "$0")"
CXX=${CXX:-nvc++}
ACC=${ACC:--acc=gpu}
printf "%5s %12s %12s  %s\n" NARR bytes/array bytes/gang placed
for NARR in 1 2 4; do
  for NFLOAT in 128 256 384 512 640 768 896 1024 1280 1536 2048 3072 4096; do
    if $CXX -O3 -std=c++17 $ACC -Minfo=acc -DNARR=$NARR -DNFLOAT=$NFLOAT -c -o /dev/null smem_limit.cpp 2>&1 \
         | grep -q 'CUDA shared memory used for a0'; then
      placed=yes
    else
      placed=no
    fi
    printf "%5d %12d %12d  %s\n" $NARR $((NFLOAT*4)) $((NARR*NFLOAT*4)) $placed
  done
done
