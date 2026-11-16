#!/bin/bash
# run_emu.sh -- run ../vadd_genx.cpp on the CPU with the CM SDK emulator.
#
#   run_emu.sh [workdir]        (CM_RT_PLATFORM=tgllp|skl|icllp, default tgllp)
#
# 1. The kernel source is compiled with the HOST compiler into a shared
#    library. A wrapper adds EXPORT_SIGNATURE(vadd) (from <shim_support.h>),
#    which the shim needs to find the kernel and its argument types --
#    without it clCreateKernel fails with -46 ("kernel address couldn't be
#    determined"). The kernel file itself stays unchanged.
# 2. The host links against libshim + libigfxcmrt_emu + libcm instead of
#    libOpenCL, and passes the library's bytes to clCreateProgramWithBinary.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# The emulator libraries come only with the 2021 CM SDK; CSDK_DIR points at it.
if [ -z "${CSDK_DIR:-}" ] || [ ! -f "$CSDK_DIR/usr/lib/libcm.so" ]; then
  echo "emu skipped: set CSDK_DIR to an unpacked 2021 CM SDK (libcm, libshim)"
  exit 0
fi
inc="-I$CSDK_DIR/usr/include/libcm -I$CSDK_DIR/usr/include/shim"
lib="-L$CSDK_DIR/usr/lib -Wl,-rpath -Wl,$CSDK_DIR/usr/lib"
work=${1:-cm_emu_run}
mkdir -p "$work"
cd "$work"

printf '#include "%s"\n#include <shim_support.h>\nEXPORT_SIGNATURE(vadd);\n' \
  "$here/../vadd_genx.cpp" > vadd_emu_kernel.cpp
g++ -std=gnu++17 -O1 -DCMRT_EMU -DSHIM -shared -fpic $inc $lib \
  vadd_emu_kernel.cpp -o vadd.shim -lcm

g++ -std=gnu++17 -O1 -DKERNEL_LIB=\"vadd.shim\" -DCL_TARGET_OPENCL_VERSION=220 $inc $lib \
  "$here/host_emu.cpp" -o host_emu -lshim -ligfxcmrt_emu -lcm -Wl,--disable-new-dtags

CM_RT_PLATFORM=${CM_RT_PLATFORM:-tgllp} ./host_emu 2>&1 | grep -v '^EMU: \[config'
