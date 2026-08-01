# plot_bk.gp -- render GDoF/s and GB/s vs DoFs from a run_benchmarks.sh file.
#
# The data file (one per kernel) holds one gnuplot `index` block per degree,
# so gnuplot reads the datasets directly; a one-line inline awk call recovers
# the legend metadata (key = p|nq, and the list of degrees) from the header.
#
# Usage:
#   gnuplot -c scripts/plot_bk.gp <datafile> [format] [outdir]
#
#   <datafile>  results/<kernel>.dat produced by run_benchmarks.sh
#   [format]    pngcairo (default) or svg
#   [outdir]    directory for the figures (default: the datafile's directory)
#
# Produces, under <outdir>:
#   <kernel>_gdofs.<ext>   throughput          GDoF/s vs ndof
#   <kernel>_gbs.<ext>     effective bandwidth  GB/s  vs ndof

if (ARGC < 1) {
    print "usage: gnuplot -c plot_bk.gp <datafile> [pngcairo|svg] [outdir]"
    exit
}

datafile = ARG1
fmt      = (ARGC >= 2 && strlen(ARG2) > 0) ? ARG2 : "pngcairo"
outdir   = (ARGC >= 3 && strlen(ARG3) > 0) ? ARG3 \
                                           : system(sprintf("dirname '%s'", datafile))
base     = system(sprintf("basename '%s' .dat", datafile))

# Make the shared style helper findable regardless of the current directory:
# prepend this script's own directory to the loadpath ($GNUPLOT_LIB is still
# searched as well).
set loadpath system(sprintf("dirname '%s'", ARG0))
load 'bk_style.gp'

# Output terminal / file extension.
if (fmt eq "svg") {
    ext = "svg"
    set terminal svg size 960,640 font 'Helvetica,13'
} else {
    ext = "png"
    set terminal pngcairo size 960,640 font ',12'
}

# Recover legend metadata from the file header with inline awk.
key  = system(sprintf("awk -F'= *' '/^# key/     {print $2; exit}' '%s'", datafile))
degs = system(sprintf("awk -F'= *' '/^# degrees/ {print $2; exit}' '%s'", datafile))
n    = words(degs)

set output sprintf("%s/%s_gdofs.%s", outdir, base, ext)
set ylabel "Throughput  [GDoF/s]"
set title base . " -- sum-factorised operator throughput"
plot for [i=1:n] datafile index (i-1) using 1:3 with linespoints pt 7 ps 0.8 \
     title sprintf("%s = %s", key, word(degs, i))

set output sprintf("%s/%s_gbs.%s", outdir, base, ext)
set ylabel "Effective bandwidth  [GB/s]"
set title base . " -- effective memory bandwidth"
plot for [i=1:n] datafile index (i-1) using 1:4 with linespoints pt 7 ps 0.8 \
     title sprintf("%s = %s", key, word(degs, i))

unset output
print sprintf("Wrote %s/%s_gdofs.%s", outdir, base, ext)
print sprintf("Wrote %s/%s_gbs.%s",   outdir, base, ext)
