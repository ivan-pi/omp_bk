#!/bin/bash
# run_cmrt.sh -- build and run the CM runtime example (hello_world.cpp).
#
#   run_cmrt.sh [emu|hw|both] [workdir]       (default: both)
#
# emu  CPU emulation: kernel + host compiled by g++ with -DCMRT_EMU, linked
#      against the SDK's libigfxcmrt_emu + libcm. Works with the SDK alone;
#      va_stub/ stands in for libva's va.h if libva-dev is not installed.
# hw   Real GPU: cmc compiles the kernel to a CM-runtime ISA ("CISA" vISA
#      container); the host links against the hardware CM runtime libigfxcmrt
#      (Intel media driver; Ubuntu packages libigfxcmrt7 + libigfxcmrt-dev +
#      intel-media-va-driver[-non-free]). The 2021 cmc must use the SDK's own
#      libocloc for this output (the system ocloc fails: "oclocInvoke failed").
#      CMRT_ISA_CPU picks the cmc target (default TGLLP = Gen12LP).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# Both modes need the 2021 CM SDK: the emulator libraries, and its cmc for
# the CISA .isa (a current cmc writes only ELF: -binary-format ocl|ze, which
# CmDevice::LoadProgram rejects with -16).
if [ -z "${CSDK_DIR:-}" ] || [ ! -x "$CSDK_DIR/usr/bin/cmc" ]; then
  echo "cmrt skipped: set CSDK_DIR to an unpacked 2021 CM SDK"
  exit 0
fi
mode=${1:-both}
work=${2:-cm_cmrt_run}
mkdir -p "$work"
cd "$work" || exit 1
status=0

if [ "$mode" = emu ] || [ "$mode" = both ]; then
  va_inc="-I$here/va_stub"
  [ -f /usr/include/va/va.h ] && va_inc=""
  g++ -std=gnu++17 -O1 -DCMRT_EMU -I"$here" $va_inc \
    -I"$CSDK_DIR/usr/include/igfxcmrt_emu" -I"$CSDK_DIR/usr/include/libcm" \
    -L"$CSDK_DIR/usr/lib" -Wl,-rpath -Wl,"$CSDK_DIR/usr/lib" \
    "$here/hello_world.cpp" -o hello_world_emu -ligfxcmrt_emu -lcm || status=1
  CM_RT_PLATFORM=${CM_RT_PLATFORM:-tgllp} ./hello_world_emu 2>&1 |
    grep -v '^EMU: \[' | tail -3
  [ "${PIPESTATUS[0]}" = 0 ] || status=1
fi

if [ "$mode" = hw ] || [ "$mode" = both ]; then
  cmrt_inc=$(dirname "$(find /usr/include -name cm_rt.h -path '*igfxcmrt*' 2>/dev/null | head -1)" 2>/dev/null)
  # (no `grep -q` here: with pipefail its early exit makes the pipeline fail)
  if [ -z "$cmrt_inc" ] || [ "$cmrt_inc" = . ] || [ -z "$(ldconfig -p | grep libigfxcmrt)" ]; then
    echo "[hw] skipped: hardware CM runtime not installed (libigfxcmrt7 + libigfxcmrt-dev)"
  else
    LD_LIBRARY_PATH="$CSDK_DIR/usr/local/lib:$CSDK_DIR/usr/lib" \
      "$CSDK_DIR/usr/bin/cmc" -mcpu="${CMRT_ISA_CPU:-TGLLP}" -m64 \
      -o hello_world_genx.isa -- "$here/hello_world_genx.cpp" >/dev/null || status=1
    g++ -std=gnu++17 -O1 -I"$here" -I"$cmrt_inc" "$here/hello_world.cpp" \
      -o hello_world_hw -ligfxcmrt -lva || status=1
    ./hello_world_hw 2>&1 | tail -3
    [ "${PIPESTATUS[0]}" = 0 ] || status=1
  fi
fi
exit $status
