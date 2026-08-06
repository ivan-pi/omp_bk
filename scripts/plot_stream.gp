# plot_stream.gp -- plot effective bandwidth vs array size from bkstream output.
#
# The bkstream executable writes a gnuplot data file with one `index` block per
# streaming kernel (init/copy/triad/striad); a one-line inline awk call recovers
# the kernel names from the header for the legend.
#
# Usage:
#   bkstream ... > results/bkstream.dat
#   gnuplot -c scripts/plot_stream.gp <datafile> [format] [outdir]
#
#   <datafile>  bkstream output (columns: nelmt ndof bytes time_s gbytes_per_s)
#   [format]    pngcairo (default) or svg
#   [outdir]    directory for the figure (default: the datafile's directory)
#
# Produces <outdir>/<base>_bw.<ext> -- effective bandwidth (GB/s) vs array size.

if (ARGC < 1) {
    print "usage: gnuplot -c plot_stream.gp <datafile> [pngcairo|svg] [outdir]"
    exit
}

datafile = ARG1
fmt      = (ARGC >= 2 && strlen(ARG2) > 0) ? ARG2 : "pngcairo"
outdir   = (ARGC >= 3 && strlen(ARG3) > 0) ? ARG3 \
                                           : system(sprintf("dirname '%s'", datafile))
base     = system(sprintf("basename '%s' .dat", datafile))

# Shared styling (logscale x, grid, key, ...) via the loadpath, which also
# searches $GNUPLOT_LIB; then override the x-axis label for a size sweep.
set loadpath system(sprintf("dirname '%s'", ARG0))
load 'bk_style.gp'
set xlabel "Array size  [bytes]"

if (fmt eq "svg") {
    ext = "svg"
    set terminal svg size 960,640 font 'Helvetica,13'
} else {
    ext = "png"
    set terminal pngcairo size 960,640 font ',12'
}

# Legend: the kernel names, in file order, from the header.
kernels = system(sprintf("awk -F'= *' '/^# kernels/ {print $2; exit}' '%s'", datafile))
n       = words(kernels)

set output sprintf("%s/%s_bw.%s", outdir, base, ext)
set ylabel "Effective bandwidth  [GB/s]"
set title base . " -- streaming bandwidth (BK DoF layout)"
plot for [i=1:n] datafile index (i-1) using 3:5 with linespoints pt 7 ps 0.8 \
     title word(kernels, i)

unset output
print sprintf("Wrote %s/%s_bw.%s", outdir, base, ext)
