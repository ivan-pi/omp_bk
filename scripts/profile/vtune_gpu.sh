#!/bin/bash
# vtune_gpu.sh -- VTune gpu-hotspots on a binary of $BUILD_DIR, one row of GPU
# metrics per kernel.  Needs /proc/sys/dev/i915/perf_stream_paranoid = 0.
#
# usage:  scripts/profile/vtune_gpu.sh <name> [args...]     e.g. BK_KERNEL=planeV1 vtune_gpu.sh bk1_offload_4
# env:    VT_MODE=overview|global-memory-accesses|compute-extended|full-compute
#         VT_FILTER=regex   kernels to print (default: all)
# The result directory is $BUILD_DIR/vt_<name>_<mode>.

. "$(dirname "$0")/../common.sh"
name=$1
shift
args=("$@")
mode=${VT_MODE:-overview}
R=$BUILD_DIR/vt_${name}_$mode
rm -rf "$R"
oneapi_env
set --
. "$ONEAPI_ROOT/vtune/latest/env/vars.sh" > /dev/null 2>&1
export OMP_TARGET_OFFLOAD=MANDATORY
capped vtune -collect gpu-hotspots -allow-multiple-runs -knob characterization-mode="$mode" -r "$R" \
    -- "$BUILD_DIR/$name" "${args[@]}" > "$R.log" 2>&1 || { tail -5 "$R.log"; exit 1; }
vtune -report hotspots -r "$R" -group-by computing-task -format csv -csv-delimiter tab > "$R.tsv" 2> /dev/null
python3 - "$R.tsv" "${VT_FILTER:-.}" <<'PY'
import sys, csv, re
rows = list(csv.reader(open(sys.argv[1]), delimiter="\t"))
h = rows[0]
for r in rows[1:]:
    if not re.search(sys.argv[2], r[0]) or "Symbol_Table" in r[0]:
        continue
    print("==", re.sub(r"\(anonymous namespace\)::|\$omp\$offloading:\d+|, int>|float, ", "", r[0]))
    for k, v in zip(h, r):
        if v not in ("", "0", "0.0") and k != "Computing Task":
            print("   %-58s %s" % (k, v))
PY
