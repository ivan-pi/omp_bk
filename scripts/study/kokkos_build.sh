#!/bin/bash
# kokkos_build.sh -- build the Kokkos CEED_BK benchmarks of dealii-X/benchmarks
# for scripts/study/kokkos_compare.sh: Kokkos with the SYCL backend, ahead of
# time for DG2 (Kokkos_ARCH_INTEL_DG2), and the BK1, BK3 and BK5
# templated_kokkos_benchmark drivers in float (they declare double, which the
# A770 lacks in hardware; nothing else is changed).
#
# usage:  scripts/study/kokkos_build.sh
#         -> $KOKKOS_DIR/ceed-build/BK{1,3,5}/templated_kokkos_benchmark
# env:    KOKKOS_DIR=<repo>/build/kokkos   sources, build trees, install
#         KOKKOS_TAG=5.3.0                 kokkos/kokkos release
#         CEED_COMMIT=457863403de6         dealii-X/benchmarks commit
#         compiles run one at a time
set -e
. "$(dirname "$0")/../common.sh"
KOKKOS_DIR=$(realpath -m "${KOKKOS_DIR:-$REPO/build/kokkos}")
KOKKOS_TAG=${KOKKOS_TAG:-5.3.0}
CEED_COMMIT=${CEED_COMMIT:-457863403de6}
oneapi_env
mkdir -p "$KOKKOS_DIR"
cd "$KOKKOS_DIR"

if [ ! -d kokkos-src ]; then
    git clone -q --depth 1 --branch "$KOKKOS_TAG" https://github.com/kokkos/kokkos.git kokkos-src
fi
if [ ! -d benchmarks ]; then
    git clone -q https://github.com/dealii-X/benchmarks.git benchmarks
    git -C benchmarks checkout -q "$CEED_COMMIT"
    sed -i 's/using T = double;/using T = float;   \/\/ was double: the A770 has no FP64/' \
        benchmarks/CEED_BK/src/BK{1,3,5}/templated_kokkos_benchmark.cc
fi

if [ ! -f kokkos-install/lib/cmake/Kokkos/KokkosConfig.cmake ]; then
    cmake -S kokkos-src -B kokkos-build -G Ninja -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_INSTALL_PREFIX="$KOKKOS_DIR/kokkos-install" \
        -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ARCH_INTEL_DG2=ON > kokkos-configure.log
    cmake --build kokkos-build -j 1
    cmake --install kokkos-build > /dev/null
fi

cmake -S benchmarks/CEED_BK -B ceed-build -G Ninja -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR="$KOKKOS_DIR/kokkos-install/lib/cmake/Kokkos" > ceed-configure.log
cmake --build ceed-build -j 1 --target templated_kokkos_benchmark \
    templated_kokkos_benchmark_bk3 templated_kokkos_benchmark_bk5
echo "KOKKOS_BK=$KOKKOS_DIR/ceed-build"
