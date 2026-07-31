#!/usr/bin/env bash
#
# run_benchmarks.sh -- sweep a bake-off kernel over a logarithmic range of DoFs.
#
# For a given benchmark executable (BK1, BK3 or BK5) this drives the kernel
# across a set of problem sizes chosen so that the total number of degrees of
# freedom (DoFs) is spaced logarithmically over the requested range (by default
# 1e4 .. 1e8). For each sample the number of elements is picked so that
#
#     ndof = nelmt * dofs_per_element(degree)
#
# lands as close as possible to the target DoF count. The achieved GDoF/s and
# GB/s reported by the kernel are parsed with awk and written, one row per run,
# to a per-degree data file that plot_results.sh turns into figures.
#
# Argument conventions differ between kernels (see README):
#   BK1, BK3 : first CLI argument is the polynomial order p, supported 1..8,
#              and dofs_per_element = (p+1)^3.
#   BK5      : first CLI argument is nq directly, supported 2..8, and
#              dofs_per_element = nq^3.
# The correct convention is chosen from the executable's basename.
#
# Usage:
#   scripts/run_benchmarks.sh [options] <executable> [dof_min] [dof_max] [degree]
#
# Positional arguments:
#   executable   Path to the benchmark binary (e.g. ./BK1, ./BK5).
#   dof_min      Lower bound of the DoF sweep      (default 1e4).
#   dof_max      Upper bound of the DoF sweep      (default 1e8).
#   degree       Polynomial order (BK1/BK3) or nq (BK5). When omitted the
#                full range of supported degrees is scanned.
#
# Options:
#   -n N         Number of log-spaced DoF sample points   (default 12).
#   -t NTESTS    Timing repetitions handed to the kernel  (default 5).
#   -o DIR       Directory for the .dat result files      (default results).
#   -p           Run scripts/plot_results.sh on the output when the sweep ends.
#   -h           Show this help and exit.
#
# Development tip: the full 1e4..1e8 sweep at large degrees allocates several
# GB and takes a while. For quick testing pass a smaller range and few points,
# e.g.  scripts/run_benchmarks.sh -n 5 ./BK5 1e3 1e5 4
#
set -euo pipefail

# --- defaults ---------------------------------------------------------------
npoints=12
ntests=5
outdir="results"
do_plot=0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    sed -n '2,50p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'
    exit "${1:-0}"
}

# --- option parsing ---------------------------------------------------------
while getopts ':n:t:o:ph' opt; do
    case "$opt" in
        n) npoints="$OPTARG" ;;
        t) ntests="$OPTARG" ;;
        o) outdir="$OPTARG" ;;
        p) do_plot=1 ;;
        h) usage 0 ;;
        :) echo "error: option -$OPTARG requires an argument" >&2; usage 1 ;;
        \?) echo "error: unknown option -$OPTARG" >&2; usage 1 ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -lt 1 ]]; then
    echo "error: missing benchmark executable" >&2
    usage 1
fi

exe="$1"
dof_min="${2:-1e4}"
dof_max="${3:-1e8}"
degree_arg="${4:-}"

if [[ ! -x "$exe" ]]; then
    echo "error: '$exe' is not an executable file (did you run 'make'?)" >&2
    exit 1
fi

# --- kernel-specific conventions --------------------------------------------
# Map the executable basename to:
#   mode        -- 'p'  : CLI arg is polynomial order, dpe = (arg+1)^3
#                  'nq' : CLI arg is nq,               dpe = arg^3
#   deg_lo/deg_hi -- inclusive range of supported degrees.
base="$(basename "$exe")"
base_uc="$(printf '%s' "$base" | tr '[:lower:]' '[:upper:]')"
case "$base_uc" in
    *BK5*)          mode="nq"; deg_lo=2; deg_hi=8 ;;
    *BK1*|*BK3*)    mode="p";  deg_lo=1; deg_hi=8 ;;
    *)
        echo "warning: unrecognised kernel '$base'; assuming BK1/BK3 argument" \
             "convention (polynomial order, degrees 1..8)" >&2
        mode="p"; deg_lo=1; deg_hi=8 ;;
esac

# dofs_per_element for a given degree under the active convention.
dpe() {
    local d="$1"
    if [[ "$mode" == "nq" ]]; then
        echo $(( d * d * d ))
    else
        echo $(( (d + 1) * (d + 1) * (d + 1) ))
    fi
}

# --- degree list ------------------------------------------------------------
degrees=()
if [[ -n "$degree_arg" ]]; then
    if (( degree_arg < deg_lo || degree_arg > deg_hi )); then
        echo "error: degree $degree_arg out of range for $base" \
             "(supported: $deg_lo..$deg_hi)" >&2
        exit 1
    fi
    degrees=("$degree_arg")
else
    for (( d = deg_lo; d <= deg_hi; ++d )); do
        degrees+=("$d")
    done
fi

mkdir -p "$outdir"

# --- log-spaced target DoFs -------------------------------------------------
# Emit `npoints` values spaced evenly in log10 between dof_min and dof_max.
log_targets() {
    awk -v lo="$dof_min" -v hi="$dof_max" -v n="$npoints" 'BEGIN {
        if (n < 1) { print "awk: need at least one point" > "/dev/stderr"; exit 1 }
        ln10 = log(10);
        llo = log(lo) / ln10; lhi = log(hi) / ln10;
        if (n == 1) { printf "%.0f\n", exp(llo * ln10); exit }
        for (i = 0; i < n; ++i) {
            e = llo + (lhi - llo) * i / (n - 1);
            printf "%.0f\n", exp(e * ln10);
        }
    }'
}

echo "# kernel      : $base (mode=$mode, arg is ${mode})"
echo "# DoF range   : $dof_min .. $dof_max  ($npoints log-spaced points)"
echo "# degrees     : ${degrees[*]}"
echo "# repetitions : $ntests"
echo "# output dir  : $outdir"
echo

manifest="$outdir/${base}.files"
: > "$manifest"

for deg in "${degrees[@]}"; do
    d_pe="$(dpe "$deg")"
    datafile="$outdir/${base}_deg$(printf '%02d' "$deg").dat"

    {
        echo "# kernel=$base  mode=$mode  degree=$deg  dofs_per_element=$d_pe"
        echo "# ndof  nelmt  gdof_per_s  gbytes_per_s"
    } > "$datafile"

    echo ">> $base degree $deg (dofs/element = $d_pe) -> $datafile"

    prev_nelmt=-1
    while read -r target; do
        # Choose nelmt so that nelmt*dpe is closest to the target DoF count,
        # never below 1 element.
        nelmt=$(( (target + d_pe / 2) / d_pe ))
        (( nelmt < 1 )) && nelmt=1
        # Skip duplicate element counts (common at the low end / high degree).
        (( nelmt == prev_nelmt )) && continue
        prev_nelmt=$nelmt

        # Run the kernel and pull the two rates out of its report with awk.
        out="$("$exe" "$deg" "$nelmt" "$ntests")"
        parsed="$(printf '%s\n' "$out" | awk '
            /GDoF\/s/ {
                for (i = 1; i <= NF; ++i) {
                    if ($i == "nelmt")  n = $(i + 2);
                    if ($i == "GDoF/s") g = $(i + 2);
                    if ($i == "GB/s")   b = $(i + 2);
                }
                print n, g, b;
            }')"

        if [[ -z "$parsed" ]]; then
            echo "   warning: could not parse output for nelmt=$nelmt; skipping" >&2
            continue
        fi

        read -r r_nelmt r_gdof r_gbs <<< "$parsed"
        ndof=$(( r_nelmt * d_pe ))
        printf '%d %d %s %s\n' "$ndof" "$r_nelmt" "$r_gdof" "$r_gbs" >> "$datafile"
        printf '   nelmt=%-10d ndof=%-12d GDoF/s=%-10s GB/s=%s\n' \
            "$r_nelmt" "$ndof" "$r_gdof" "$r_gbs"
    done < <(log_targets)

    echo "$datafile" >> "$manifest"
done

echo
echo "Wrote data files listed in $manifest"

if [[ "$do_plot" == "1" ]]; then
    echo
    "$script_dir/plot_results.sh" -o "$outdir" "$exe"
fi
