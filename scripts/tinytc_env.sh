# tinytc_env.sh -- put a tinytc install and the oneAPI runtime it needs into
# the current bash shell, for the tinytc and tinytc-opt tools and for
# configuring CMake.  tinytc ships no environment script of its own.
#
# usage:  . scripts/tinytc_env.sh          (source it; bash only)
#         tinytc-opt -pcheck-ir kernel.ir
#         tinytc -d tgl -o kernel.spv kernel.ir
# env:    TINYTC_ROOT=/opt/tinytc/latest    the install (bin/, lib/, lib/cmake/)
#         ONEAPI_ROOT, ICPX_VERSION         as in common.sh
#
# Sets TINYTC_ROOT and prepends to PATH, LD_LIBRARY_PATH and
# CMAKE_PREFIX_PATH.  A tinytc built with icpx links oneAPI libraries
# (libsvml), so even the command-line tools need the oneAPI runtime.
# bk_tinytc itself needs none of this: the a770 preset passes TINYTC_ROOT
# and the binary finds libtinytc through its RPATH.

if [ -z "${BASH_VERSION:-}" ] || [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "tinytc_env.sh: source it from bash: . scripts/tinytc_env.sh" >&2
    exit 1
fi

_tinytc_env() {
    local root=${TINYTC_ROOT:-/opt/tinytc/latest}
    if [ ! -d "$root/lib/cmake/tinytc" ]; then
        echo "tinytc_env.sh: no tinytc install at $root (set TINYTC_ROOT)" >&2
        return 1
    fi
    . "$(dirname "${BASH_SOURCE[0]}")/common.sh"
    oneapi_env
    TINYTC_ROOT=$(cd "$root" && pwd -P)
    export TINYTC_ROOT
    local var dir
    for var in PATH:bin LD_LIBRARY_PATH:lib CMAKE_PREFIX_PATH:; do
        dir=$TINYTC_ROOT/${var#*:}
        dir=${dir%/}
        var=${var%%:*}
        case ":${!var:-}:" in
            *":$dir:"*) ;;
            *) export "$var=$dir${!var:+:${!var}}" ;;
        esac
    done
    echo "tinytc $(basename "$TINYTC_ROOT") from $TINYTC_ROOT"
}
_tinytc_env
