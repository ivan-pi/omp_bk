#!/bin/bash
# run.sh -- build cache_worker.cpp with and without the cache directive and run
# every n in its own process (a device fault aborts the process).
# usage: ./run.sh [CXX=nvc++] ; env CXX, ACC override the compiler and its flags
cd "$(dirname "$0")"
CXX=${CXX:-nvc++}
case $CXX in
  *nvc*) ACC=${ACC:--acc=gpu} ;;
  *)     ACC=${ACC:--fopenacc} ;;
esac
set -e
$CXX -O3 -std=c++17 $ACC -o cache_worker cache_worker.cpp
$CXX -O3 -std=c++17 $ACC -DNO_CACHE -o cache_worker_nocache cache_worker.cpp
set +e
for exe in ./cache_worker_nocache ./cache_worker; do
  echo "== $exe"
  for n in 3 4 5 6 7 8 9 10; do
    $exe $n || echo "n = $n: exit status $?"
  done
done
