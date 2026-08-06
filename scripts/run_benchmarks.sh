#!/usr/bin/env bash
#
# run_benchmarks.sh -- sweep a bake-off kernel over a logarithmic range of DoFs.
#
# Runs BK1/BK3/BK5 across problem sizes whose total DoF count is spaced
# logarithmically over [dof_min, dof_max] (default 1e4 .. 1e8). For each sample
# the element count is chosen so that ndof = nelmt * (p+1)^3 lands closest to
# the target. Results go into a single gnuplot data file <outdir>/<kernel>.dat,
# one `index` block per polynomial order (see scripts/plot_bk.gp). Collecting
# and plotting are separate steps.
#
# The logarithmic DoF targets come from the companion `scripts/logspace` tool
# (build everything with `make`); this keeps the script free of awk-dialect
# quirks and portable wherever a C++ compiler exists.
#
# Usage:
#   scripts/run_benchmarks.sh [options] <executable> [dof_min] [dof_max] [degree]
#
# Positional arguments:
#   executable   Path to the benchmark binary (e.g. ./BK1, ./BK5).
#   dof_min      Lower bound of the DoF sweep      (default 1e4).
#   dof_max      Upper bound of the DoF sweep      (default 1e8).
#   degree       Polynomial order p (1..8). When omitted p = 1..8 is scanned.
#
# Options:
#   -n N         Number of log-spaced DoF sample points   (default 12).
#   -t NTESTS    Timing repetitions handed to the kernel  (default 5).
#   -o DIR       Directory for the .dat result file        (default results).
#   -h           Show this help and exit.
#
set -euo pipefail

npoints=12
ntests=5
outdir="results"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
logspace="$script_dir/logspace"

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'; exit "${1:-0}"; }

while getopts ':n:t:o:h' opt; do
    case "$opt" in
        n) npoints="$OPTARG" ;;
        t) ntests="$OPTARG" ;;
        o) outdir="$OPTARG" ;;
        h) usage 0 ;;
        :)  echo "error: option -$OPTARG requires an argument" >&2; usage 1 ;;
        \?) echo "error: unknown option -$OPTARG" >&2; usage 1 ;;
    esac
done
shift $((OPTIND - 1))

[[ $# -ge 1 ]] || { echo "error: missing benchmark executable" >&2; usage 1; }
exe="$1"; dof_min="${2:-1e4}"; dof_max="${3:-1e8}"; degree_arg="${4:-}"
base="$(basename "$exe")"

[[ -x "$exe" ]]      || { echo "error: '$exe' is not executable (run 'make'?)" >&2; exit 1; }
[[ -x "$logspace" ]] || { echo "error: '$logspace' missing; build it with 'make'" >&2; exit 1; }

# All kernels take polynomial order p (1..8); DoFs per element = (p+1)^3.
p_lo=1; p_hi=8
dpe() { echo $(( ($1 + 1) * ($1 + 1) * ($1 + 1) )); }

# Degrees to scan: the given one, or the full supported range.
if [[ -n "$degree_arg" ]]; then
    (( degree_arg >= p_lo && degree_arg <= p_hi )) \
        || { echo "error: degree $degree_arg out of range (supported: $p_lo..$p_hi)" >&2; exit 1; }
    degrees=("$degree_arg")
else
    degrees=(); for (( d = p_lo; d <= p_hi; ++d )); do degrees+=("$d"); done
fi

mkdir -p "$outdir"
datafile="$outdir/${base}.dat"
echo "# $base: p = ${degrees[*]} | DoFs $dof_min..$dof_max ($npoints pts) | $ntests reps -> $datafile"

# Log-spaced DoF targets (same for every degree), as a space-separated list.
targets="$("$logspace" "$dof_min" "$dof_max" "$npoints")"

# Data-file header (metadata consumed by plot_bk.gp).
{
    echo "# kernel = $base"
    echo "# key = p"
    echo "# degrees = ${degrees[*]}"
    echo "# columns: ndof  nelmt  gdof_per_s  gbytes_per_s"
} > "$datafile"

first=1
for deg in "${degrees[@]}"; do
    d_pe="$(dpe "$deg")"
    echo ">> p=$deg (dofs/element = $d_pe)"

    # Two blank lines separate gnuplot `index` blocks; then the block comment.
    { (( first )) || printf '\n\n'
      printf '\n# p = %s  (dofs_per_element=%s)\n' "$deg" "$d_pe"; } >> "$datafile"
    first=0

    prev=-1
    for target in $targets; do
        nelmt=$(( (target + d_pe / 2) / d_pe ))   # nearest element count
        (( nelmt < 1 )) && nelmt=1
        (( nelmt == prev )) && continue           # skip duplicates (low end / high p)
        prev=$nelmt

        # awk extracts the (floating-point) rate fields; the ndof integer
        # arithmetic below stays in the shell, whose 64-bit ints avoid the
        # overflow mawk's 32-bit %d would hit on large sweeps.
        out="$("$exe" "$deg" "$nelmt" "$ntests")"
        parsed="$(printf '%s\n' "$out" | awk '/GDoF\/s/{
            for (i=1;i<=NF;++i) {
                if ($i=="nelmt")  n=$(i+2)
                if ($i=="GDoF/s") g=$(i+2)
                if ($i=="GB/s")   b=$(i+2)
            }
            print n, g, b; exit }')"
        [[ -n "$parsed" ]] \
            || { echo "error: cannot parse output of '$exe $deg $nelmt': $out" >&2; exit 1; }

        read -r r_nelmt r_gdof r_gbs <<< "$parsed"
        ndof=$(( r_nelmt * d_pe ))
        printf '%d %d %s %s\n' "$ndof" "$r_nelmt" "$r_gdof" "$r_gbs" >> "$datafile"
        printf '   nelmt=%-10d ndof=%-12d GDoF/s=%-10s GB/s=%s\n' \
            "$r_nelmt" "$ndof" "$r_gdof" "$r_gbs"
    done
done

echo "Wrote $datafile"
echo "Plot with: gnuplot -c scripts/plot_bk.gp $datafile"
