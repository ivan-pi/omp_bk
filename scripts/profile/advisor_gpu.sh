#!/bin/bash
# advisor_gpu.sh -- Intel Advisor GPU roofline (Survey + Trip Counts & FLOP
# on the GPU) of a binary of $BUILD_DIR, then the per-kernel survey as CSV and
# the interactive roofline as HTML.  Needs /proc/sys/dev/i915/perf_stream_paranoid = 0.
#
# usage:  scripts/profile/advisor_gpu.sh <name> [args...]
#         e.g. BK_KERNEL="pt6f_b1 sf6c_b1x2" advisor_gpu.sh bk5_offload_7 29155 3
# The project directory is $BUILD_DIR/adv_<name>[_$ADV_TAG]; the reports are
# <project>.csv (survey, GPU kernels) and <project>.html (roofline).
. "$(dirname "$0")/../common.sh"
name=$1
shift
args=("$@")
P=$BUILD_DIR/adv_${name}${ADV_TAG:+_$ADV_TAG}
rm -rf "$P"
oneapi_env
set --
. "$ONEAPI_ROOT/advisor/latest/env/vars.sh" > /dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
capped advisor --collect=roofline --profile-gpu --project-dir="$P" \
    -- "$BUILD_DIR/$name" "${args[@]}" > "$P.log" 2>&1 || { tail -15 "$P.log"; exit 1; }
advisor --report=survey --gpu --project-dir="$P" --format=csv --report-output="$P.csv" >> "$P.log" 2>&1
advisor --report=roofline --gpu --project-dir="$P" --report-output="$P.html" >> "$P.log" 2>&1
echo "project $P"
echo "survey  $P.csv"
echo "roofline $P.html"
