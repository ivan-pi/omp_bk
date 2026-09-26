#!/bin/bash
# high_order.sh -- throughput sweep for the high orders p = 9..14 (nq = 11..16,
# up to 4096 quadrature points per element), which only the AMX binary
# supports; writes results_high.csv in the same format as throughput.sh so the
# two files can be joined for the plots:
#
#   amx/scripts/throughput.sh results.csv
#   amx/scripts/high_order.sh results_high.csv
#   tail -n +2 results_high.csv >> results.csv
#   amx/scripts/plot_throughput.py results.csv
#
# usage: ./high_order.sh [results_high.csv]
# env:   ORDERS="9 10 11 12 13 14", DOF_MIN/DOF_MAX/PPD/NTESTS/OMP_NUM_THREADS as
#        in throughput.sh; KERNELS defaults to the kernels the AMX binary runs
#        (the BK1 executable stops at p = 8).
ORDERS=${ORDERS:-"9 10 11 12 13 14"} \
KERNELS=${KERNELS:-"serial omp_v neon_aos neon_soa amx_aos amx_soa"} \
exec "$(dirname "$0")/throughput.sh" "${1:-results_high.csv}"
