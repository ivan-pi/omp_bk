#!/bin/bash
# install_toolchain.sh -- user-space install (no sudo; the system GPU driver
# is left alone) of the C for Metal compiler and the ocloc it was released
# with, from Intel's GPU package repository.  cm/Makefile uses it by default.
#
#   cm/install_toolchain.sh [-b build] [-c cm_version] [prefix]
#
#   -b build    driver build the compiler is paired with (default 1146:
#               compute-runtime 25.18, IGC 2.11)
#   -c version  intel-igc-cm version (default 1.0.225.54093-1136~24.04)
#   prefix      default ~/opt/intel-cm-<build>; the Makefile's CM_TOOLS is
#               <prefix>/bin
#
# The pairing: the SPIR-V of a cmc (package intel-igc-cm) is read only by the
# IGC of its own driver release (a newer IGC fails with "verify Module
# failed"), so the kernels are finalized by that release's ocloc.  The
# resulting binaries run on the newer system driver.
#
# Installs <prefix>/bin/cmc and <prefix>/bin/ocloc-cm, wrappers that give
# only these two commands the matched libraries.  Packages already in
# <prefix>/debs (and its Packages index) are reused, not downloaded again.
set -euo pipefail

build=1146
cmver=1.0.225.54093-1136~24.04
while getopts "b:c:" opt; do
    case $opt in
        b) build=$OPTARG ;;
        c) cmver=$OPTARG ;;
        *) sed -n '2,20p' "$0"; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
prefix=$(realpath -m "${1:-$HOME/opt/intel-cm-$build}")
repo=https://repositories.intel.com/gpu/ubuntu

mkdir -p "$prefix/debs" "$prefix/root" "$prefix/bin"
cd "$prefix/debs"

# package -> pool path, from the repository index
[ -s Packages ] || curl -sfL "$repo/dists/noble/unified/binary-amd64/Packages.gz" | zcat > Packages
pick() { # pick <package> <version regex>
    awk -v p="$1" -v v="$2" '/^Package:/{n=$2} /^Version:/{ver=$2}
        /^Filename:/{if (n == p && ver ~ v) f=$2} END{print f}' Packages
}
files=("$(pick intel-igc-cm "^${cmver//./\\.}\$")")
for p in libigc2 libigdfcl2 intel-ocloc; do
    files+=("$(pick "$p" "-$build~")")
done
for f in "${files[@]}"; do
    if [ -z "$f" ]; then
        echo "package not found in the index (build $build, intel-igc-cm $cmver)" >&2
        exit 1
    fi
    [ -f "$(basename "$f")" ] || curl -sfLO "$repo/$f"
    dpkg-deb -x "$(basename "$f")" "$prefix/root"
done

# the soname links ldconfig would create on a real install
lib=$prefix/root/usr/lib/x86_64-linux-gnu
(
    cd "$lib"
    for f in libigc.so.2.*+0 libiga64.so.2.*+0 libigdfcl.so.2.*+0; do
        ln -sf "$f" "${f%%.so.*}.so.2"
    done
)

# wrappers: the matched libraries are seen by these two commands only
ocl=$(ls "$prefix"/root/usr/bin/ocloc-*)
printf '#!/bin/sh\nLD_LIBRARY_PATH=%s exec %s "$@"\n' "$lib" "$prefix/root/usr/bin/cmc" > "$prefix/bin/cmc"
printf '#!/bin/sh\nLD_LIBRARY_PATH=%s exec %s "$@"\n' "$lib" "$ocl" > "$prefix/bin/ocloc-cm"
chmod +x "$prefix/bin/cmc" "$prefix/bin/ocloc-cm"

"$prefix/bin/cmc" --version | grep -a 'clang version'
echo "ocloc: $(basename "$ocl")"
echo "installed in $prefix (make -C cm CM_TOOLS=$prefix/bin, the default for ~/opt/intel-cm-1146)"
