#!/bin/bash
# build.sh -- build the reproducer (icpx with SYCL, tinytc from /opt/tinytc/latest)
set -e
cd "$(dirname "$0")"
TT=${TINYTC_ROOT:-/opt/tinytc/latest}
set --
. /opt/intel/oneapi/setvars.sh > /dev/null
icpx -fsycl -O2 -std=c++17 -I"$TT/include" repro.cpp -o repro -L"$TT/lib" -Wl,-rpath,"$TT/lib" -ltinytc_sycl -ltinytc
echo "built $(pwd)/repro"
