#!/bin/bash
# build.sh -- build the C for Metal BK1 kernel (one binary per order) and its
# OpenCL host driver into $BUILD_DIR (default build/cm).
#
# usage:  cm/build.sh [p...]         default: 1 2 3 4 5 6 7 8
#   -> build/cm/bk1_cm (host) and build/cm/bk1_cm_<p>.bin (kernel, nq = p + 2)
#
# Kernel: cmc (the 2021 CM SDK, $CSDK_DIR) -> SPIR-V -> ocloc for $CM_GPU
# (default dg2) with the vector-compute backend.  From p = 4 (nq = 6) the
# kernel uses the large register file (-double-GRF: 256 GRFs, half the
# threads per EU) to hold its padded boxes without spills; GRF=128 or
# GRF=256 forces one mode for every order.  CMC_FLAGS adds cmc options
# (e.g. -DSTORE_LANES=32).  EPT=n compiles n elements per thread into
# bk1_cm_<p>_e<n>.bin (run it with BK_EPT=n).  KEEP=1 keeps the IGC dumps (Gen asm) in
# $BUILD_DIR/dump_<p>/.
#
# Host: g++ with OpenMP (the reference) and the OpenCL ICD loader.

set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR=$(realpath -m "${BUILD_DIR:-$REPO/build/cm}")
CSDK_DIR=${CSDK_DIR:-/opt/intel/cm_sdk_20211028}
CMC=${CMC:-$CSDK_DIR/usr/bin/cmc}
CM_GPU=${CM_GPU:-dg2}
CXX=${CXX:-g++}
orders=${*:-1 2 3 4 5 6 7 8}
mkdir -p "$BUILD_DIR"

for p in $orders; do
    nq=$((p + 2))
    grf=${GRF:-$([ "$nq" -ge 6 ] && echo 256 || echo 128)}
    opts="-vc-codegen -ftranslate-legacy-memory-intrinsics"
    [ "$grf" = 256 ] && opts="$opts -double-GRF"
    ept=${EPT:-1}
    name=bk1_cm_$p$([ "$ept" = 1 ] || echo "_e$ept")
    spv=$BUILD_DIR/$name.spv
    "$CMC" -emit-spirv -fcmocl -mcpu=TGLLP -m64 -DNQ=$nq -DEPT=$ept ${CMC_FLAGS:-} -o "$spv" -- "$REPO/cm/bk1_cm_genx.cpp"
    dump=()
    if [ "${KEEP:-0}" = 1 ]; then
        rm -rf "$BUILD_DIR/dump_$p"
        mkdir -p "$BUILD_DIR/dump_$p"
        dump=(env IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$BUILD_DIR/dump_$p")
    fi
    log=$BUILD_DIR/$name.log
    ${dump[@]+"${dump[@]}"} ocloc compile -q -file "$spv" -spirv_input -device "$CM_GPU" -options "$opts" \
        -output "$name" -output_no_suffix -out_dir "$BUILD_DIR" > "$log" 2>&1 || { cat "$log"; exit 1; }
    spill=$(sed -n 's/.*Spill memory used = \([0-9]*\) bytes for kernel bk1_cm.*/\1/p' "$log")
    echo "$name.bin  nq=$nq GRF=$grf${spill:+  SPILL=${spill}B}"
done

"$CXX" -std=c++17 -O2 -fopenmp -Wall -I"$REPO/include" -I"$REPO/cm" "$REPO/cm/bk1_cm.cpp" -lOpenCL \
    -o "$BUILD_DIR/bk1_cm"
echo "bk1_cm (host)"
