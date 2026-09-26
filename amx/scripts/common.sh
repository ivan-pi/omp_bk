# common.sh -- shared by the scripts in this directory; source it with
#   . "$(dirname "$0")/common.sh"
#
# Binaries (all overridable from the environment, paths relative to the cwd):
BK1_AMX=${BK1_AMX:-./bk1_amx}     # amx/BK1_amx.cpp
BK1=${BK1:-./bk1}                 # BK1.cpp built without OpenMP (serial reference)
BK1_OMP=${BK1_OMP:-./bk1_omp}     # BK1.cpp built with -fopenmp (host fallback)

# Parse the driver's "... GDoF/s = <g> GB/s = <b>" line from stdin.
rates() { sed -n 's/.*GDoF\/s = \([0-9.e+-]*\) GB\/s = \([0-9.e+-]*\).*/\1 \2/p'; }   # "<g> <b>"
gdofs() { sed -n 's/.*GDoF\/s = \([0-9.e+-]*\).*/\1/p'; }                                # "<g>"

# Elements for a target number of degrees of freedom at order p (at least 1).
nelmt_for_dofs() { awk -v d="$1" -v p="$2" 'BEGIN { n = int(d / ((p+1)^3) + 0.5); print (n < 1 ? 1 : n) }'; }
