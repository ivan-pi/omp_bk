#!/bin/bash
# run_examples.sh -- build every CM example of this skill and run its checks.
#
#   run_examples.sh [-d device] [-g name] [-s] [workdir]
#
#   -d device  ocloc device for the binaries (default dg2; set it to your GPU's,
#              e.g. rkl, tgllp, mtl-h, bmg: see references/targets.md)
#   -g name    substring of the GPU name to run on (sets CM_DEVICE; default: keep
#              $CM_DEVICE if set, else the Intel GPU with the most compute units)
#   -s         also run the vadd host on SYCL's OpenCL backend (default: Level Zero)
#   workdir    scratch directory (default: ./cm_examples_run)
#
# Hosts are SYCL (cm_sycl.hpp; needs the oneAPI environment for icpx) except
# the OpenCL C++ ones for image surfaces (linear), spec constants (specconst)
# and the plain CLHPP reference (host_ocl). ICPX overrides the compiler
# command, e.g. a wrapper: ICPX="<wrapper> icpx".
# Each test prints PASS/FAIL; the script exits non-zero if any failed.
#
# Toolchain: the current cmc + matched ocloc-cm after
# `source ~/opt/intel-cm-<build>/env.sh` (install_cm_toolchain.sh); .spv and
# CM-source runs then go through `cm_run` (the matched OpenCL / Level Zero
# drivers), .bin runs use the system driver, and the LSC, spec-constant and
# "-cmc" source examples are added. Without it: the 2021 SDK cmc + system
# ocloc, from CSDK_DIR (an unpacked 2021 CM SDK). The CPU-emulation and CM
# runtime examples need CSDK_DIR in either case and are skipped without it.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
ex=$here/examples
build=$here/scripts/cm_build.sh
device=dg2
gpu=
sycl=0
while getopts "d:g:s" opt; do
  case $opt in
    d) device=$OPTARG ;;
    g) gpu=$OPTARG ;;
    s) sycl=1 ;;
    *) sed -n '2,25p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
work=${1:-cm_examples_run}
mkdir -p "$work" && cd "$work" || exit 1
if [ -n "$gpu" ]; then
  export CM_DEVICE=$gpu
fi
fail=0
spv=() # prefix for runs that finalize SPIR-V at run time
if command -v ocloc-cm >/dev/null && command -v cm_run >/dev/null; then
  spv=(cm_run)
  echo "toolchain: $(cmc --version 2>&1 | grep -a 'clang version' | head -1)"
fi
if [ ${#spv[@]} = 0 ] && [ -z "${CSDK_DIR:-}" ] && [ -z "${CMC:-}" ]; then
  echo "no CM compiler: source ~/opt/intel-cm-<build>/env.sh, or set CSDK_DIR to a 2021 CM SDK"
  exit 1
fi

run() { # run <label> <command...>
  local out
  out=$("${@:2}" 2>&1)
  local rc=$?
  echo "$out" | grep -E 'PASS|FAIL|device:|skipped|from the GPU|compiled for' | sed "s/^/[$1] /"
  if [ $rc -ne 0 ]; then
    echo "[$1] exit $rc"
    echo "$out" | tail -5
    fail=1
  fi
}

# Kernels: name, cmc -mcpu. dpas only for devices with XMX units.
kernels="vadd:TGLLP patterns:TGLLP linear:TGLLP spmv:TGLLP argmax:TGLLP scan:TGLLP gesv:TGLLP gesvb:TGLLP traits:TGLLP target:TGLLP"
case $device in
  dg2|acm*|bmg|lnl|pvc|ptl|arl|arl-h) kernels="$kernels dpas:XEHP_SDV" ;;
esac
for k in $kernels; do
  name=${k%%:*}
  cpu=${k#*:}
  "$build" -d "$device" -m "$cpu" -o out "$ex/${name}_genx.cpp" >/dev/null 2>build_$name.log ||
    { echo "[$name] build failed, see $work/build_$name.log"; fail=1; }
done

# Hosts: SYCL by default (one icpx at a time), OpenCL C++ where SYCL can't pass the arguments.
read -r -a icpx <<<"${ICPX:-icpx}"
command -v "${icpx[0]}" >/dev/null || { echo "icpx not found: source /opt/intel/oneapi/setvars.sh"; exit 1; }
sycl_host() { # sycl_host <name> [extra flags...]
  "${icpx[@]}" -fsycl -O2 -fp-model=precise -I"$ex" "${@:2}" "$ex/host_$1.cpp" -lze_loader -lOpenCL \
    -o "host_$1" >build_host_$1.log 2>&1 || { echo "[host_$1] build failed, see $work/build_host_$1.log"; fail=1; }
}
for h in sycl patterns spmv argmax scan gesv traits target dpas; do
  sycl_host $h
done
# batch-in-lanes solver: kernel default N=4, 1 rhs -> a matching host build.
"${icpx[@]}" -fsycl -O2 -fp-model=precise -I"$ex" -DGESV_N=4 -DGESV_NRHS=1 "$ex/host_gesv.cpp" \
  -lze_loader -lOpenCL -o host_gesvb >build_host_gesvb.log 2>&1 || fail=1
for h in ocl linear_ocl; do
  g++ -std=c++17 -O2 "$ex/host_$h.cpp" -lOpenCL -o "host_$h" || fail=1
done

run vadd.spv ${spv[@]+"${spv[@]}"} ./host_sycl out/vadd_genx.spv
run vadd.bin ./host_sycl out/vadd_genx.bin
run vadd.clhpp ${spv[@]+"${spv[@]}"} ./host_ocl out/vadd_genx.spv
run patterns.spv ${spv[@]+"${spv[@]}"} ./host_patterns out/patterns_genx.spv
run patterns.bin ./host_patterns out/patterns_genx.bin
run linear.spv ${spv[@]+"${spv[@]}"} ./host_linear_ocl out/linear_genx.spv
run linear.bin ./host_linear_ocl out/linear_genx.bin
run spmv.spv ${spv[@]+"${spv[@]}"} ./host_spmv out/spmv_genx.spv
run spmv.bin ./host_spmv out/spmv_genx.bin
run argmax.bin ./host_argmax out/argmax_genx.bin 26
run scan.bin ./host_scan out/scan_genx.bin 24
run gesv.bin ./host_gesv out/gesv_genx.bin 16384
run gesvb.bin ./host_gesvb out/gesvb_genx.bin 100003 lanes
run traits.spv ${spv[@]+"${spv[@]}"} ./host_traits out/traits_genx.spv
run target.spv ${spv[@]+"${spv[@]}"} ./host_target out/target_genx.spv
run target.bin ./host_target out/target_genx.bin
case $kernels in
  *dpas*) run dpas.bin ./host_dpas out/dpas_genx.bin ;;
esac
# Current cmc only: specialization constants (the 2021 cmc segfaults on
# them), the LSC API (no cm_lsc.h in 2021; LSC GPUs only) and run-time CM
# source builds ("-cmc", needs the matched front end + IGC).
if [ ${#spv[@]} -gt 0 ]; then
  if "$build" -d "$device" -o out "$ex/specconst_genx.cpp" >/dev/null 2>build_specconst.log &&
    g++ -std=c++17 -O2 "$ex/host_specconst_ocl.cpp" -lOpenCL -o host_specconst_ocl; then
    run specconst.spv cm_run ./host_specconst_ocl out/specconst_genx.spv 8 2.5
    run specconst.bin ./host_specconst_ocl out/specconst_genx.bin 4 0
  else
    echo "[specconst] build failed, see $work/build_specconst.log"
    fail=1
  fi
  case $device in
    dg2|acm*|bmg|lnl|pvc|ptl|arl*|mtl*)
      sycl_host lsc
      if "$build" -d "$device" -o out "$ex/lsc_genx.cpp" >/dev/null 2>build_lsc.log; then
        run lsc.spv cm_run ./host_lsc out/lsc_genx.spv
        run lsc.bin ./host_lsc out/lsc_genx.bin
        run lsc.cmc cm_run ./host_lsc "$ex/lsc_genx.cpp"
      else
        echo "[lsc] build failed, see $work/build_lsc.log"
        fail=1
      fi
      ;;
  esac
  run vadd.cmc cm_run ./host_sycl "$ex/vadd_genx.cpp"
  run vadd.clhpp.cmc cm_run ./host_ocl "$ex/vadd_genx.cpp"
fi

# CPU emulation of vadd (SDK shim layer; no GPU involved).
run emu.cpu "$ex/emu/run_emu.sh" emu
# CM runtime example: emulated, plus the GPU if libigfxcmrt is installed.
run cmrt "$ex/cmrt/run_cmrt.sh" both cmrt

if [ "$sycl" = 1 ]; then
  # The same SYCL host on the OpenCL backend: SPIR-V, binary (interop) and CM source.
  run sycl.ocl.spv ${spv[@]+"${spv[@]}"} env ONEAPI_DEVICE_SELECTOR=opencl:gpu ./host_sycl out/vadd_genx.spv
  run sycl.ocl.bin env ONEAPI_DEVICE_SELECTOR=opencl:gpu ./host_sycl out/vadd_genx.bin
  if [ ${#spv[@]} -gt 0 ]; then
    run sycl.ocl.cmc cm_run env ONEAPI_DEVICE_SELECTOR=opencl:gpu ./host_sycl "$ex/vadd_genx.cpp"
  fi
fi

[ $fail = 0 ] && echo "ALL EXAMPLES PASS" || echo "SOME EXAMPLES FAILED"
exit $fail
