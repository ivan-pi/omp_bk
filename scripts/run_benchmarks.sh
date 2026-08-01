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
# lands as close as possible to the target DoF count.
#
# Results for one kernel go into a single column-format data file,
#   <outdir>/<kernel>.dat
# with one dataset per degree, datasets separated by two blank lines so that
# gnuplot can address them directly via `index` (see scripts/plot_bk.gp). This
# script only *collects* data; plotting is a separate step.
#
# Every kernel takes the polynomial order p (1..8) as its first CLI argument
# and carries (p+1)^3 DoFs per element: BK1/BK3 use nq = p + 2 quadrature
# points (nm = p + 1 modes), and BK5 is collocated with nq = p + 1.
#
# Usage:
#   scripts/run_benchmarks.sh [options] <executable> [dof_min] [dof_max] [degree]
#
# Positional arguments:
#   executable   Path to the benchmark binary (e.g. ./BK1, ./BK5).
#   dof_min      Lower bound of the DoF sweep      (default 1e4).
#   dof_max      Upper bound of the DoF sweep      (default 1e8).
#   degree       Polynomial order p (1..8). When omitted the full supported
#                range p = 1..8 is scanned.
#
# Options:
#   -n N         Number of log-spaced DoF sample points   (default 12).
#   -t NTESTS    Timing repetitions handed to the kernel  (default 5).
#   -o DIR       Directory for the .dat result file        (default results).
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

usage() {
    sed -n '2,43p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'
    exit "${1:-0}"
}

# --- option parsing ---------------------------------------------------------
while getopts ':n:t:o:h' opt; do
    case "$opt" in
        n) npoints="$OPTARG" ;;
        t) ntests="$OPTARG" ;;
        o) outdir="$OPTARG" ;;
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

# --- argument convention ----------------------------------------------------
# All three kernels take the polynomial order p as their first argument, over
# p = 1..8, and carry (p+1)^3 DoFs per element (see the header above).
base="$(basename "$exe")"
mode="p"        # legend key recorded in the data file
deg_lo=1
deg_hi=8

# dofs_per_element for polynomial order p.
dpe() {
    local d="$1"
    echo $(( (d + 1) * (d + 1) * (d + 1) ))
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

# Scratch space for per-degree data blocks; assembled into the final file only
# once each block is known to be non-empty.
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# --- log-spaced target DoFs -------------------------------------------------
# Emit `npoints` values spaced evenly in log10 between dof_min and dof_max.
# (mawk lacks the ** operator, so exponentiate via exp/log.)
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

echo "# kernel      : $base (first arg = polynomial order p)"
echo "# DoF range   : $dof_min .. $dof_max  ($npoints log-spaced points)"
echo "# degrees     : ${degrees[*]}"
echo "# repetitions : $ntests"
echo "# output dir  : $outdir"
echo

# --- run the sweep, one block per degree ------------------------------------
committed_degs=()   # degrees that produced at least one data row
committed_dpe=()

for deg in "${degrees[@]}"; do
    d_pe="$(dpe "$deg")"
    block="$tmpdir/block_$deg"
    : > "$block"

    echo ">> $base $mode=$deg (dofs/element = $d_pe)"

    prev_nelmt=-1
    while read -r target; do
        # Choose nelmt so that nelmt*dpe is closest to the target DoF count,
        # never below 1 element.
        nelmt=$(( (target + d_pe / 2) / d_pe ))
        (( nelmt < 1 )) && nelmt=1
        # Skip duplicate element counts (common at the low end / high degree).
        (( nelmt == prev_nelmt )) && continue
        prev_nelmt=$nelmt

        # Run the kernel and pull nelmt + the two rates out of its report.
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
        printf '%d %d %s %s\n' "$ndof" "$r_nelmt" "$r_gdof" "$r_gbs" >> "$block"
        printf '   nelmt=%-10d ndof=%-12d GDoF/s=%-10s GB/s=%s\n' \
            "$r_nelmt" "$ndof" "$r_gdof" "$r_gbs"
    done < <(log_targets)

    if [[ -s "$block" ]]; then
        committed_degs+=("$deg")
        committed_dpe+=("$d_pe")
    else
        echo "   warning: no data collected for $mode=$deg; omitting" >&2
    fi
done

if [[ ${#committed_degs[@]} -eq 0 ]]; then
    echo "error: no data collected for any degree" >&2
    exit 1
fi

# --- assemble the single per-kernel data file -------------------------------
# Column format with one gnuplot `index` block per degree, blocks separated by
# two blank lines. A short metadata header lets the plotting script recover the
# legend labels with a one-line awk call.
datafile="$outdir/${base}.dat"
{
    echo "# kernel = $base"
    echo "# key = $mode"
    echo "# degrees = ${committed_degs[*]}"
    echo "# columns: ndof  nelmt  gdof_per_s  gbytes_per_s"

    for i in "${!committed_degs[@]}"; do
        deg="${committed_degs[$i]}"
        d_pe="${committed_dpe[$i]}"
        # Two blank lines separate datasets (gnuplot `index` boundary).
        (( i > 0 )) && printf '\n\n'
        printf '\n# %s = %s  (dofs_per_element=%s)\n' "$mode" "$deg" "$d_pe"
        cat "$tmpdir/block_$deg"
    done
} > "$datafile"

echo
echo "Wrote $datafile (${#committed_degs[@]} degree dataset(s))"
echo "Plot it with:  gnuplot -c scripts/plot_bk.gp $datafile"
