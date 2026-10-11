#!/bin/sh
# cuda_headers_glibc241.sh -- a patched copy of a CUDA toolkit's include
# tree for glibc 2.41 and later.
#
# glibc 2.41 declares cospi, sinpi, rsqrt and their float forms in math.h
# with noexcept(true); the toolkits up to 12.8 declare them in
# crt/math_functions.h without an exception specification (13.x still
# lacks it on rsqrt), and nvcc rejects the second declaration:
#   error: exception specification is incompatible with that of previous
#   function "cospi" (declared at line ... of .../crt/math_functions.h)
# The distributions patch the toolkit's header; this script applies the
# same change to a copy of the whole include tree in a directory you own,
# which nvcc then searches before the toolkit's own (user include paths
# come first), so every header resolves in the copy.
#
# Usage:  cuda/cuda_headers_glibc241.sh <toolkit include dir> <destination>
#   e.g.  cuda/cuda_headers_glibc241.sh $NVHPC_ROOT/cuda/12.8/include ~/cuda-12.8-include
#   then  make BK1_ff_cuda_plain NVCC=$NVHPC_ROOT/cuda/12.8/bin/nvcc NVCCEXTRA="-I$HOME/cuda-12.8-include"
# The script prints the patched declarations for a check.

set -eu
if [ $# -ne 2 ]; then
    echo "usage: $0 <toolkit include dir> <destination>" >&2
    exit 2
fi
src=$1
dst=$2
if [ ! -f "$src/crt/math_functions.h" ]; then
    echo "no crt/math_functions.h under $src" >&2
    exit 1
fi
mkdir -p "$dst"
cp -R "$src/." "$dst/"
f="$dst/crt/math_functions.h"
# declarations `extern ... name(args);` of the six functions that carry no
# exception specification yet
sed -i -E '/noexcept|__THROW|throw[[:space:]]*\(/! s/^(extern .*[^A-Za-z0-9_](cospi|cospif|sinpi|sinpif|rsqrt|rsqrtf)\([^;]*\))[[:space:]]*;/\1 noexcept(true);/' "$f"
echo "patched declarations in $f:"
grep -nE '^extern .*[^A-Za-z0-9_](cospi|cospif|sinpi|sinpif|rsqrt|rsqrtf)\(' "$f" || true
echo "build with: NVCCEXTRA=\"-I$dst\""
