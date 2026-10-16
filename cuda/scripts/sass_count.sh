#!/usr/bin/env bash
# sass_count.sh -- instructions per kernel in the SASS of one or more
# binaries (cuobjdump -sass), demangled with cu++filt when it is on the PATH.
# Code size is the direct measure of what each compiler emits per phase and
# of how far it unrolled.  Works on nvcc executables and on nvc++ OpenMP
# offload executables (the device code is embedded in both).
#
#   cuda/scripts/sass_count.sh BK1_ff_cuda_plain BK1_ff
#   cuda/scripts/sass_count.sh -a BK1_ff          # every kernel, not only the BK1 ones
set -euo pipefail

all=0
if [ "${1:-}" = "-a" ]; then
    all=1
    shift
fi
if [ $# -eq 0 ]; then
    echo "usage: $0 [-a] binary..." >&2
    exit 2
fi
filt=cat
if command -v cu++filt >/dev/null 2>&1; then
    filt=cu++filt
fi
for bin in "$@"; do
    echo "== $bin"
    cuobjdump -sass "$bin" \
    | awk '/Function : /{f=$3} /^[[:space:]]+\/\*[0-9a-f]+\*\//{n[f]++} END{for (k in n) print n[k], k}' \
    | if [ $all -eq 1 ]; then cat; else grep -i 'MassOperatorFF\|SumFactorization'; fi \
    | while read -r count name; do
        printf '%8d  %s\n' "$count" "$(echo "$name" | $filt | sed 's/(unsigned int.*//; s/(unsigned long.*//')"
      done \
    | sort -k2
done
