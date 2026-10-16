#!/usr/bin/env bash
# regs.sh -- registers, stack frame and spills per kernel from a ptxas
# report on stdin: nvcc's -Xptxas -v or nvc++'s -gpu=ptxinfo.  Only fixed
# strings are matched.  Kernel names are demangled with cu++filt when it is
# on the PATH (nvc++'s nvkernel_ names stay as they are; their readable part
# is the SumFactorizationFF...ILi<nq>ELi<batch>E in the middle).
#
#   make -B BK1_ff_cuda_plain NVCCEXTRA='-Xptxas -v' 2>&1 | cuda/scripts/regs.sh
#   nvc++ ... -gpu=ptxinfo -c BK1_ff.cpp 2>&1 | cuda/scripts/regs.sh
set -uo pipefail
filt=cat
if command -v cu++filt >/dev/null 2>&1; then
    filt=cu++filt
fi
printf '%5s %6s %8s %8s  %s\n' regs stack spill_st spill_ld kernel
awk '
    index($0, "Compiling entry function") { name = $(NF-2); gsub(/\x27/, "", name) }
    index($0, "bytes stack frame")        { stack = $1; st = $5; ld = $9 }
    index($0, "Used") && index($0, "registers") {
        for (i = 1; i <= NF; ++i) { if ($i == "Used") { regs = $(i+1) } }
        print regs, stack, st, ld, name
        stack = st = ld = ""
    }' \
| while read -r regs stack st ld name; do
    printf '%5s %6s %8s %8s  %s\n' "$regs" "$stack" "$st" "$ld" "$(echo "$name" | $filt | sed 's/(unsigned.*//')"
  done \
| sort -k5
