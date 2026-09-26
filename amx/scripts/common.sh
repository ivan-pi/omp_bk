# common.sh -- shared by the scripts in this directory; source it with
#   . "$(dirname "$0")/common.sh"
#
# Binaries: the Makefile's outputs in the repository root (make BK1 BK1_amx),
# overridable from the environment.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BK1_AMX=${BK1_AMX:-$ROOT/BK1_amx}   # amx/BK1_amx.cpp: AMX, NEON and reference kernels
BK1=${BK1:-$ROOT/BK1}               # BK1.cpp (OpenMP target); only validate.sh's norm cross-check uses it

# Parse the driver's "... GDoF/s = <g> GB/s = <b>" line from stdin.
rates() { sed -n 's/.*GDoF\/s = \([0-9.e+-]*\) GB\/s = \([0-9.e+-]*\).*/\1 \2/p'; }   # "<g> <b>"
gdofs() { sed -n 's/.*GDoF\/s = \([0-9.e+-]*\).*/\1/p'; }                                # "<g>"

# Elements for a target number of degrees of freedom at order p (at least 1).
nelmt_for_dofs() { awk -v d="$1" -v p="$2" 'BEGIN { n = int(d / ((p+1)^3) + 0.5); print (n < 1 ? 1 : n) }'; }
