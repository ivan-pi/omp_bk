#!/usr/bin/env python3
"""plot_roofline.py -- put the BK1 kernels against the two roofs.

usage: ./plot_roofline.py results.csv bw.csv [out.png] [--ghz 3.5] [--pcores 6] [--rfo]
                          [--amx-store GB/s] [--title "..."]
       (--ghz/--pcores are only used if bw.csv has no fma_peak row)

The plateau per kernel and order is the best GDoF/s among the three largest
problem sizes that were run (the serial kernel is capped at 1e7 DoF by
throughput.sh, the others go to 1e8).

Left panel:  per order, the measured plateau of each kernel (best GDoF/s over
             sizes >= 1e7 DoF) against
               memory roof  = BW / bytes-per-DoF(p)         (BW = "add" from bw.csv:
                                                              2 reads + 1 write, the BK1 mix)
               NEON FMA roof = pcores * ghz * 16 / MAC-per-DoF(p)
Right panel: the classic roofline, GFLOP/s vs arithmetic intensity (flop/byte),
             one point per (kernel, order), with the memory and FMA roofs.

bytes-per-DoF counts in + out + JxW; --rfo adds 4 bytes/DoF for the
write-allocate of `out` (use it if bw_test shows "fill" well below "read").
"""
import sys
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

opts = sys.argv[1:]
# positional arguments: everything that is neither an option nor an option's value
option_values = {opts[i + 1] for i, a in enumerate(opts[:-1]) if a in ("--ghz", "--pcores", "--amx-store", "--title")}
args = [a for a in opts if not a.startswith("--") and a not in option_values]
res_csv = args[0] if len(args) > 0 else "results.csv"
bw_csv = args[1] if len(args) > 1 else "bw.csv"
dst = args[2] if len(args) > 2 else "roofline.png"
ghz = float(opts[opts.index("--ghz") + 1]) if "--ghz" in opts else 3.5
pcores = int(opts[opts.index("--pcores") + 1]) if "--pcores" in opts else 6
rfo = "--rfo" in opts
# AMX store-path roof: aggregate store bandwidth of the units in GB/s.  From
# amx_pipe on M2 Pro a 128-byte pair store costs ~1.2 ns (~32 B/cycle), i.e.
# ~107 GB/s per unit; default assumes two P-cluster units.
amx_store_gbs = float(opts[opts.index("--amx-store") + 1]) if "--amx-store" in opts else 214.0
title = opts[opts.index("--title") + 1] if "--title" in opts else "BK1: measured plateaus vs memory and FMA roofs"
PLATEAU_TOP = 3     # plateau = best of the three largest sizes run for that kernel and order

# --- bandwidth -------------------------------------------------------------
bw = {}
with open(bw_csv) as f:
    for row in csv.reader(f):
        if len(row) >= 5 and row[0] != "test":
            bw[row[0]] = float(row[4])
bw_roof = bw.get("add", bw.get("triad", max(v for k, v in bw.items() if k != "fma_peak")))
bw_read = bw.get("read", bw.get("dot", bw_roof))
# compute roof: measured (fma_peak row, GFLOP/s at the real clock) if present,
# else pcores * ghz * 16 lanes * 2 flop
peak = bw["fma_peak"] if "fma_peak" in bw else pcores * ghz * 16.0 * 2.0
peak_label = (f"measured FMA peak {peak:.0f} GFLOP/s" if "fma_peak" in bw
              else f"{pcores} P-cores @ {ghz} GHz = {peak:.0f} GFLOP/s")
amx_peak = bw.get("amx_peak")          # fma32 outer-product peak, GFLOP/s, full tiles

# --- measured plateaus -----------------------------------------------------
points = defaultdict(lambda: defaultdict(list))   # points[kernel][p] -> [(dofs, gdofs)]
with open(res_csv) as f:
    for row in csv.DictReader(f):
        try:
            g = float(row["gdofs"])
        except ValueError:
            continue
        if math.isnan(g):
            continue
        points[row["kernel"]][int(row["p"])].append((int(row["dofs"]), g))
plateau = defaultdict(dict)   # plateau[kernel][p] = best of the PLATEAU_TOP largest sizes
for k, byp in points.items():
    for p, pts in byp.items():
        top = sorted(pts)[-PLATEAU_TOP:]
        plateau[k][p] = max(g for _, g in top)

labels = {"serial": "serial", "omp": "OpenMP (scalar)", "omp_v": "OpenMP, loops interchanged",
          "neon_aos": "NEON, element-major", "neon_soa": "NEON, elements-on-lanes",
          "amx_aos": "AMX, element-major", "amx_soa": "AMX, elements-on-lanes"}
order = [k for k in ["serial", "omp", "omp_v", "neon_aos", "neon_soa", "amx_aos", "amx_soa"] if k in plateau]
orders = sorted({p for k in plateau.values() for p in k})

def nm(p): return p + 1
def nq(p): return p + 2
def bytes_per_dof(p):
    b = 4.0 * (2 * nm(p) ** 3 + nq(p) ** 3) / nm(p) ** 3
    return b + (4.0 if rfo else 0.0)
def mac_per_dof(p):
    T = nm(p) ** 2 + nm(p) * nq(p) + nq(p) ** 2
    return 2.0 * nq(p) * T / nm(p) ** 2

# AMX: each fma32 is a full 16x16 outer product but only N of the 16 rows are
# useful (N = nq forward, nm reverse); averaged over the six steps weighted by
# the number of outer products, the useful fraction is nm*nq / (8*(nm+nq)).
def amx_fill(p): return nm(p) * nq(p) / (8.0 * (nm(p) + nq(p)))
# bytes the six sum-factorized stages store per DoF (every intermediate is
# written once through the unit's store path)
def store_bytes_per_dof(p):
    a, b = nm(p), nq(p)
    return 4.0 * (a ** 3 + 2 * a * a * b + 2 * a * b * b + b ** 3) / a ** 3

mem_roof = {p: bw_roof / bytes_per_dof(p) for p in orders}          # GDoF/s
fma_roof = {p: peak / (2.0 * mac_per_dof(p)) for p in orders}         # GDoF/s

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- left: per-order ceilings ---------------------------------------------
ax1.plot(orders, [mem_roof[p] for p in orders], "k-", lw=2,
         label=f"memory roof ({bw_roof:.0f} GB/s{', +RFO' if rfo else ''})")
ax1.plot(orders, [fma_roof[p] for p in orders], "k--", lw=2,
         label=f"NEON FMA roof ({peak_label})")
if amx_peak:
    ax1.plot(orders, [amx_peak * amx_fill(p) / (2.0 * mac_per_dof(p)) for p in orders], "k-.", lw=1.2,
             label=f"AMX FMA roof: fma32 peak {amx_peak:.0f} GFLOP/s x row fill N/16")
ax1.plot(orders, [amx_store_gbs / store_bytes_per_dof(p) for p in orders], "k:", lw=2,
         label=f"AMX store roof: {amx_store_gbs:.0f} GB/s / stored bytes per DoF")
for i, k in enumerate(order):
    ps = sorted(plateau[k])
    ax1.plot(ps, [plateau[k][p] for p in ps], "o-", ms=4, color=f"C{i}", label=labels.get(k, k))
ax1.set_xlabel("polynomial order p")
ax1.set_ylabel(f"GDoF/s (plateau: best of the {PLATEAU_TOP} largest sizes)")
ax1.set_title("throughput against the per-order ceilings")
ax1.set_xticks(orders)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

# --- right: classic roofline ---------------------------------------------
# Compute limits are horizontal (independent of intensity), bandwidth limits
# are diagonals of slope one.  The per-order AMX ceilings (fill, store path)
# depend on p and are therefore only drawn on the left panel.  The AMX
# kernels are plotted twice: useful (algorithmic) flops, and raw flops
# (useful / row fill) as hollow markers, to be read against the raw AMX peak.
ai = {p: 2 * mac_per_dof(p) / bytes_per_dof(p) for p in orders}
ai_min = min(ai.values()) / 3
ai_max = max(ai.values()) * 3
xs = [ai_min * (ai_max / ai_min) ** (i / 100) for i in range(101)]
ax2.plot(xs, [bw_roof * x for x in xs], "k-", lw=2, label=f"DRAM roof, {bw_roof:.0f} GB/s (add)")
ax2.plot(xs, [bw_read * x for x in xs], "k:", lw=1, label=f"DRAM read-only, {bw_read:.0f} GB/s")
ax2.axhline(peak, color="k", ls="--", lw=2, label=f"NEON FMA peak, {peak:.0f} GFLOP/s")
ymax = peak * 1.5
if amx_peak:
    ax2.axhline(amx_peak, color="k", ls="-.", lw=1.5, label=f"AMX fma32 peak (full tiles), {amx_peak:.0f} GFLOP/s")
    ymax = amx_peak * 1.5
for i, k in enumerate(order):
    ps = sorted(plateau[k])
    ax2.plot([ai[p] for p in ps], [plateau[k][p] * 2 * mac_per_dof(p) for p in ps],
             "o-", ms=4, color=f"C{i}", label=labels.get(k, k))
    if k.startswith("amx"):
        ax2.plot([ai[p] for p in ps], [plateau[k][p] * 2 * mac_per_dof(p) / amx_fill(p) for p in ps],
                 "o--", ms=5, mfc="none", color=f"C{i}", lw=0.8, label=f"{labels.get(k, k)}, raw flops (/ fill)")
    for p in ps:
        if p in (orders[0], orders[-1]):
            ax2.annotate(f"p={p}", (ai[p], plateau[k][p] * 2 * mac_per_dof(p)),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(ai_min, ai_max)
ax2.set_ylim(top=ymax)
ax2.set_xlabel("arithmetic intensity (flop / byte; sum-factorized flops, compulsory DRAM bytes)")
ax2.set_ylabel("GFLOP/s")
ax2.set_title("roofline (horizontal: compute peaks, diagonal: DRAM)")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=7, loc="lower right")

fig.suptitle(title)
fig.tight_layout()
fig.savefig(dst, dpi=150, bbox_inches="tight")
print(f"wrote {dst}")
print(f"bandwidth roof {bw_roof:.1f} GB/s (add), read-only {bw_read:.1f} GB/s, compute roof {peak_label}"
      + (f", AMX peak {amx_peak:.0f} GFLOP/s" if amx_peak else ""))
print(f"{'p':>3} {'B/DoF':>7} {'MAC/DoF':>8} {'stB/DoF':>8} {'mem roof':>9} {'FMA roof':>9} {'AMXst roof':>10}  " + "  ".join(f"{labels.get(k,k)[:14]:>14}" for k in order))
for p in orders:
    print(f"{p:>3} {bytes_per_dof(p):>7.1f} {mac_per_dof(p):>8.1f} {store_bytes_per_dof(p):>8.1f} {mem_roof[p]:>9.2f} {fma_roof[p]:>9.2f} {amx_store_gbs / store_bytes_per_dof(p):>10.2f}  "
          + "  ".join((f"{plateau[k][p]:>14.2f}" if p in plateau[k] else f"{'-':>14}") for k in order))
