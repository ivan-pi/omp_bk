# bk_style.gp -- shared axis/key styling for the bake-off sweep figures.
#
# Loaded by plot_bk.gp through the gnuplot loadpath, which plot_bk.gp sets to
# its own directory and which additionally searches $GNUPLOT_LIB. To use it
# standalone from another location, put this file on GNUPLOT_LIB, e.g.
#
#     export GNUPLOT_LIB=/path/to/omp_bk/scripts
#
# and `load 'bk_style.gp'` from your own gnuplot script.

set logscale x
set grid xtics ytics mxtics
set xlabel "Number of DoFs"
set format x "10^{%L}"
set key top left
set tics nomirror
