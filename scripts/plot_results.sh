#!/usr/bin/env bash
#
# plot_results.sh -- post-process run_benchmarks.sh output into figures.
#
# Reads the per-degree .dat files produced by run_benchmarks.sh for one kernel
# and, using awk to marshal the data and gnuplot to render, produces two
# figures plotted against the number of DoFs (log x-axis):
#
#   <base>_gdofs.<ext>  -- throughput   GDoF/s vs ndof
#   <base>_gbs.<ext>    -- bandwidth    GB/s   vs ndof
#
# One curve per polynomial degree, labelled with the kernel's own argument
# convention (p = ... for BK1/BK3, nq = ... for BK5).
#
# Usage:
#   scripts/plot_results.sh [-o DIR] [-f FORMAT] <executable-or-base>
#
#   -o DIR       Directory holding the .dat files / receiving figures
#                (default results).
#   -f FORMAT    Output format: pngcairo (default) or svg.
#   <base>       Benchmark executable or its basename (e.g. ./BK5 or BK5); used
#                to locate DIR/<base>_deg*.dat.
#
set -euo pipefail

outdir="results"
format="pngcairo"

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'
    exit "${1:-0}"
}

while getopts ':o:f:h' opt; do
    case "$opt" in
        o) outdir="$OPTARG" ;;
        f) format="$OPTARG" ;;
        h) usage 0 ;;
        :) echo "error: option -$OPTARG requires an argument" >&2; usage 1 ;;
        \?) echo "error: unknown option -$OPTARG" >&2; usage 1 ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -lt 1 ]]; then
    echo "error: missing benchmark name" >&2
    usage 1
fi
base="$(basename "$1")"

case "$format" in
    pngcairo) ext="png" ;;
    svg)      ext="svg" ;;
    *) echo "error: unsupported format '$format' (use pngcairo or svg)" >&2; exit 1 ;;
esac

# Collect the per-degree data files in ascending degree order.
shopt -s nullglob
files=("$outdir/${base}_deg"*.dat)
shopt -u nullglob
if [[ ${#files[@]} -eq 0 ]]; then
    echo "error: no data files matching $outdir/${base}_deg*.dat" >&2
    echo "       run scripts/run_benchmarks.sh first" >&2
    exit 1
fi

# Read the legend label prefix (p vs nq) from a data file header.
label="$(awk -F'mode=' '/^# kernel=/ { split($2, a, " "); print a[1]; exit }' "${files[0]}")"
[[ "$label" == "nq" ]] && key="nq" || key="p"

# Build the per-curve plot fragments. awk extracts the degree recorded in each
# file's header so the legend is correct regardless of file ordering.
build_plot() { # $1 = y column
    local col="$1" first=1 frag=""
    for f in "${files[@]}"; do
        local deg
        deg="$(awk '/^# kernel=/ { for (i = 1; i <= NF; ++i)
                    if ($i ~ /^degree=/) { split($i, a, "="); print a[2] } exit }' "$f")"
        [[ -z "$deg" ]] && deg="?"
        (( first )) || frag+=", "
        first=0
        frag+="'$f' using 1:$col with linespoints pointtype 7 pointsize 0.8 title '$key = $deg'"
    done
    printf '%s' "$frag"
}

gdofs_plot="$(build_plot 3)"
gbs_plot="$(build_plot 4)"

gdofs_out="$outdir/${base}_gdofs.$ext"
gbs_out="$outdir/${base}_gbs.$ext"

gnuplot <<GPL
set terminal $format size 960,640 $([[ "$format" == pngcairo ]] && echo "font ',12'")
set logscale x
set grid xtics ytics mxtics
set xlabel "Number of DoFs"
set key top left
set format x "10^{%L}"

set ylabel "Throughput  [GDoF/s]"
set title "$base -- sum-factorised operator throughput"
set output "$gdofs_out"
plot $gdofs_plot

set ylabel "Effective bandwidth  [GB/s]"
set title "$base -- effective memory bandwidth"
set output "$gbs_out"
plot $gbs_plot
GPL

echo "Wrote $gdofs_out"
echo "Wrote $gbs_out"
