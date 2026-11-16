#!/bin/bash
# install_cm_toolchain.sh -- user-space install (no sudo, system driver left
# alone) of a current C for Metal compiler plus the IGC, ocloc, OpenCL and
# Level Zero runtimes it was released with, from Intel's GPU apt repository.
#
#   install_cm_toolchain.sh [-b build] [-c cm_version] [prefix]
#
#   -b build    driver build to pair with (default 1146 = compute-runtime
#               25.18.33578.15 + IGC 2.11.12, the "LTS 2523" generation)
#   -c version  intel-igc-cm version (default 1.0.225.54093-1136~24.04)
#   prefix      default ~/opt/intel-cm-<build>
#
# Why the pairing: intel-igc-cm's cmc writes SPIR-V that only the IGC of its
# own release reads. A newer IGC such as 2.42 (compute-runtime 26.40)
# rejects even a plain vadd with "spirv_read_verify: verify Module failed".
#
# Afterwards `source <prefix>/env.sh`: it puts wrappers on PATH: cmc, the
# matched `ocloc-cm`, and `cm_run` (alias `cm_ocl`), a prefix that runs one
# host program on the matched OpenCL AND Level Zero runtimes
# (OCL_ICD_VENDORS, ZE_ENABLE_ALT_DRIVERS, LD_LIBRARY_PATH for it only):
#   cmc -march=dg2 -emit-spirv -o k.spv k_genx.cpp      # new front end
#   cmc -march=dg2 -o k.bin k_genx.cpp                  # straight to a zebin
#   ocloc-cm compile -file k.spv -spirv_input -device dg2 -options "-vc-codegen ..."
#   cm_run ./host k.spv        # run-time IL path, spec constants (OpenCL, L0, SYCL)
#   cm_run ./host k_genx.cpp   # run-time CM source build ("-cmc")
#   ./host k.bin               # the binary also loads on the system driver
set -euo pipefail

build=1146
cmver=1.0.225.54093-1136~24.04
while getopts "b:c:" opt; do
  case $opt in
    b) build=$OPTARG ;;
    c) cmver=$OPTARG ;;
    *) sed -n '2,27p' "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
prefix=${1:-$HOME/opt/intel-cm-$build}
repo=https://repositories.intel.com/gpu/ubuntu
index=$repo/dists/noble/unified/binary-amd64/Packages.gz

mkdir -p "$prefix/debs" "$prefix/root" "$prefix/vendors"
cd "$prefix/debs"

# Package -> pool path, from the repository index.
curl -sfL "$index" | zcat > Packages
pick() { # pick <package> <version-regex>
  awk -v p="$1" -v v="$2" '/^Package:/{n=$2} /^Version:/{ver=$2}
    /^Filename:/{if (n == p && ver ~ v) f=$2} END{print f}' Packages
}
files=("$(pick intel-igc-cm "^${cmver//./\\.}\$")")
for p in libigc2 libigdfcl2 intel-ocloc intel-opencl-icd libze-intel-gpu1; do
  files+=("$(pick $p "-$build~")")
done
for f in "${files[@]}"; do
  if [ -z "$f" ]; then
    echo "package not found in $index (build $build, cm $cmver)" >&2
    exit 1
  fi
  [ -f "$(basename "$f")" ] || curl -sfLO "$repo/$f"
  dpkg-deb -x "$(basename "$f")" "$prefix/root"
  echo "unpacked $(basename "$f")"
done

# ldconfig would create these soname links on a real install.
lib=$prefix/root/usr/lib/x86_64-linux-gnu
(
  cd "$lib"
  for f in libigc.so.2.*+0 libiga64.so.2.*+0 libigdfcl.so.2.*+0; do
    ln -sf "$f" "${f%%.so.*}.so.2"
  done
)
echo "$lib/intel-opencl/libigdrcl.so" > "$prefix/vendors/intel-cm.icd"

# Wrappers: the matched libraries are seen only by these commands, never by
# programs running on the system driver (IGC 2.11 loaded under a newer
# runtime such as 26.40 fails to build anything).
mkdir -p "$prefix/bin"
ocl=$(ls "$prefix"/root/usr/bin/ocloc-*)
printf '#!/bin/sh\nLD_LIBRARY_PATH=%s exec %s "$@"\n' "$lib" "$prefix/root/usr/bin/cmc" > "$prefix/bin/cmc"
printf '#!/bin/sh\nLD_LIBRARY_PATH=%s exec %s "$@"\n' "$lib" "$ocl" > "$prefix/bin/ocloc-cm"
{
  echo '#!/bin/sh'
  echo '# Run a host program on the matched OpenCL and Level Zero runtimes.'
  echo "OCL_ICD_VENDORS=$prefix/vendors \\"
  echo "ZE_ENABLE_ALT_DRIVERS=$lib/libze_intel_gpu.so.1 \\"
  echo "LD_LIBRARY_PATH=$lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} exec \"\$@\""
} > "$prefix/bin/cm_run"
ln -sf cm_run "$prefix/bin/cm_ocl"
chmod +x "$prefix"/bin/*
echo "export PATH=$prefix/bin:\$PATH" > "$prefix/env.sh"

"$prefix/bin/cmc" --version | grep -a clang
echo "installed in $prefix; use: source $prefix/env.sh  (adds cmc, ocloc-cm, cm_run)"
