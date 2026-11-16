#!/bin/bash
# cm_build.sh -- compile a C for Metal kernel file through every pipeline stage.
#
#   cm_build.sh [-d device] [-m cmc_cpu] [-o outdir] [-k] kernel_genx.cpp
#
#   -d device   ocloc device for the final binary (default: dg2; e.g. tgllp, rkl, bmg)
#   -m cpu      2021 SDK cmc only: -mcpu value (default: TGLLP; it knows SKL,
#               ICLLP, TGLLP, XEHP_SDV, ... but not DG2 or newer). A current
#               cmc (intel-igc-cm, clang 11+) gets -march=<device> instead.
#   -o outdir   output directory (default: ./cm_out)
#   -x opts     ocloc/backend options (default: "-vc-codegen", plus
#               "-ftranslate-legacy-memory-intrinsics" for LSC devices (DG2 and newer),
#               where legacy DWord atomics fail without it; pre-LSC devices reject it)
#   CMC_FLAGS   environment: extra cmc options, e.g. CMC_FLAGS="-DCHUNK=8192 -DU=8"
#   CMC, OCLOC  environment: the compilers. Default: the current cmc and its
#               matched ocloc-cm after `source ~/opt/intel-cm-<build>/env.sh`
#               (install_cm_toolchain.sh), because a current cmc's SPIR-V needs
#               the IGC of its own release; otherwise the 2021 SDK cmc in
#               $CSDK_DIR (no default) with the system ocloc.
#   -k          keep intermediates: preprocessed .i, front-end LLVM IR .ll,
#               SPIR-V text .spvasm, and IGC dumps (optimized IR, vISA, GEN asm)
#
# Produces <name>.spv (portable, load with -vc-codegen) and <name>.bin
# (device binary, load with clCreateProgramWithBinary / zeModuleCreate native).
set -euo pipefail

if [ -z "${CMC:-}" ]; then
  if command -v ocloc-cm >/dev/null; then
    CMC=$(command -v cmc)
  elif [ -n "${CSDK_DIR:-}" ]; then
    CMC=$CSDK_DIR/usr/bin/cmc
  else
    echo "cm_build.sh: no cmc; source ~/opt/intel-cm-<build>/env.sh, or set CSDK_DIR (2021 SDK) or CMC" >&2
    exit 1
  fi
fi
if "$CMC" --version 2>/dev/null | grep -aq "clang version 8\."; then
  legacy=1 # 2021 SDK: -mcpu, -fcmocl, -m64, "--" before the source
else
  legacy=0
fi
if [ -z "${OCLOC:-}" ]; then
  if [ $legacy = 0 ] && command -v ocloc-cm >/dev/null; then
    OCLOC=ocloc-cm
  else
    OCLOC=ocloc
  fi
fi
LLVM_SPIRV=${LLVM_SPIRV:-/opt/intel/oneapi/compiler/latest/bin/compiler/llvm-spirv}
device=dg2
cpu=TGLLP
outdir=cm_out
keep=0
vcopts=""
while getopts "d:m:o:x:k" opt; do
  case $opt in
    d) device=$OPTARG ;;
    m) cpu=$OPTARG ;;
    o) outdir=$OPTARG ;;
    x) vcopts=$OPTARG ;;
    k) keep=1 ;;
    *) sed -n '2,24p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [ -z "$vcopts" ]; then
  case $device in
    skl|kbl|cfl|glk|icllp|icl|ehl|jsl|tgllp|tgl|rkl|adls|adl-s|adlp|adl-p|adln|adl-n|dg1|gen9|gen11|gen12lp)
      vcopts="-vc-codegen" ;;
    *) vcopts="-vc-codegen -ftranslate-legacy-memory-intrinsics" ;;
  esac
fi
src=$(realpath "$1")
name=$(basename "${src%.*}")
mkdir -p "$outdir"
cd "$outdir"

# 1. CM source -> SPIR-V (SPV_INTEL_vector_compute, llvm.genx.* intrinsics)
if [ $legacy = 1 ]; then
  fe() { "$CMC" "$@" -fcmocl -mcpu="$cpu" -m64 ${CMC_FLAGS:-} -- "$src"; }
else
  fe() { "$CMC" "$@" -march="$device" ${CMC_FLAGS:-} "$src"; }
fi
fe -emit-spirv -o "$name.spv"
spv=$(pwd)/$name.spv
bindir=$(pwd)

if [ "$keep" = 1 ]; then
  fe -E -o "$name.i"
  fe -emit-llvm -S -o "$name.ll"
  [ -x "$LLVM_SPIRV" ] && "$LLVM_SPIRV" -to-text "$name.spv" -o "$name.spvasm"
  mkdir -p igc_dump
  dumpenv=(env IGC_ShaderDumpEnable=1 IGC_DumpToCurrentDir=1)
  cd igc_dump
fi

# 2. SPIR-V -> device binary with the system ocloc (vector-compute backend of IGC)
${dumpenv[@]+"${dumpenv[@]}"} "$OCLOC" compile -q -file "$spv" \
  -spirv_input -device "$device" -options "$vcopts" \
  -output "$name" -output_no_suffix -out_dir "$bindir"
cd "$bindir"
ls -l "$name".*
[ "$keep" = 1 ] && ls igc_dump | sed 's/^/  igc_dump\//'
exit 0
