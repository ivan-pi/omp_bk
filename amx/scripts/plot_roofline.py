#!/usr/bin/env python3
"""plot_roofline.py -- put the BK1 kernels against the two roofs.

usage: ./plot_roofline.py results.csv bw.csv [out.png] [--ghz 3.5] [--pcores 6] [--rfo]
                          [--amx-store GB/s] [--title "..."]
       (--ghz/--pcores are only used if bw.csv has no fma_peak row)

The plateau per kernel and order is the best GDoF/s among the three largest
problem sizes that were run (the serial kernel is capped at 1e7 DoF by
throughput.sh, the others go to 1e8).

Left panel:  per order, the measured plateau of each kernel against the
             ceilings, on a log axis so that the slow baselines and the roofs
             (which reach 16 GDoF/s at high order) fit one panel:
               memory roof  = BW / bytes-per-DoF(p)         (BW = "add" from bw.csv:
                                                              2 reads + 1 write, the BK1 mix)
               NEON FMA roof = pcores * ghz * 16 / MAC-per-DoF(p)
               AMX FMA roof and AMX store-path roof (see below)
Right panel: the classic roofline, GFLOP/s vs arithmetic intensity (flop/byte),
             one point per (kernel, order), with the memory and FMA roofs.
The roofs are labelled at their right end; the kernels share one legend below.

bytes-per-DoF counts in + out + JxW; --rfo adds 4 bytes/DoF for the
write-allocate of `out` (use it if bw_test shows "fill" well below "read").
"""
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

import bkplot as bk

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
title = opts[opts.index("--title") + 1] if "--title" in opts else "BK1 (fp32): measured plateaus vs memory and FMA roofs"

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
plateau = bk.plateaus(bk.load_results(res_csv))
kernels = bk.kernels_in(plateau)
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
amx_roof = {p: amx_peak * amx_fill(p) / (2.0 * mac_per_dof(p)) for p in orders} if amx_peak else {}
store_roof = {p: amx_store_gbs / store_bytes_per_dof(p) for p in orders}

ROOF_COLOR = "#52514e"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))


# --- left: per-order ceilings ---------------------------------------------
# The ceilings are thin, recessive lines named at their right end (not in the
# legend); names whose lines end close together are pushed apart on the log axis.
roofs = [("-", [mem_roof[p] for p in orders], f"memory roof {bw_roof:.0f} GB/s{' + RFO' if rfo else ''}"),
         ("--", [fma_roof[p] for p in orders], f"NEON FMA roof {peak:.0f} GFLOP/s"),
         (":", [store_roof[p] for p in orders], f"AMX store roof {amx_store_gbs:.0f} GB/s")]
if amx_roof:
    roofs.append(("-.", [amx_roof[p] for p in orders], f"AMX FMA roof {amx_peak:.0f} GFLOP/s x fill"))
for ls, ys, _ in roofs:
    ax1.plot(orders, ys, ls, color=ROOF_COLOR, lw=1.3)
ends = sorted((ys[-1], text) for _, ys, text in roofs)
label_y = [y for y, _ in ends]
for i in range(1, len(label_y)):                     # at least a factor 1.3 apart
    label_y[i] = max(label_y[i], label_y[i - 1] * 1.3)
for (y_end, text), y in zip(ends, label_y):
    ax1.annotate(text, (orders[-1], y_end), xytext=(orders[-1] + 0.35, y), textcoords="data",
                 fontsize=7.5, color=ROOF_COLOR, va="center",
                 arrowprops=dict(arrowstyle="-", color=ROOF_COLOR, lw=0.5) if y != y_end else None)
for k in kernels:
    ps = sorted(plateau[k])
    bk.kernel_line(ax1, k, ps, [plateau[k][p] for p in ps])
ax1.set_yscale("log")
ax1.set_xlabel("polynomial order p")
ax1.xaxis.set_label_coords((sum(orders) / len(orders) - (orders[0] - 0.3)) / (orders[-1] + 4.5 - (orders[0] - 0.3)), -0.075)
ax1.set_ylabel(f"GDoF/s (plateau: best of the {bk.PLATEAU_TOP} largest sizes)")
ax1.set_title("throughput against the per-order ceilings", fontsize=11)
ax1.set_xticks(orders)
ax1.set_xlim(orders[0] - 0.3, orders[-1] + 4.5)          # room for the roof names
ax1.grid(True, which="major", alpha=0.25)

# --- right: classic roofline ---------------------------------------------
# Compute limits are horizontal (independent of intensity), bandwidth limits
# are diagonals of slope one.  The per-order AMX ceilings (fill, store path)
# depend on p and are therefore only drawn on the left panel.  The AMX
# kernels are plotted twice: useful (algorithmic) flops, and raw flops
# (useful / row fill) as hollow markers, to be read against the raw AMX peak.
ai = {p: 2 * mac_per_dof(p) / bytes_per_dof(p) for p in orders}
ai_min = min(ai.values()) / 2
ai_max = max(ai.values()) * 2.5
xs = [ai_min * (ai_max / ai_min) ** (i / 100) for i in range(101)]
ymax = (amx_peak or peak) * 1.6
ymin = min(plateau[k][p] * 2 * mac_per_dof(p) for k in kernels for p in plateau[k]) / 2


def diagonal(gbs, ls, text, at_top):
    """A bandwidth line, named where it leaves through the top (at_top) or enters at the left."""
    ax2.plot(xs, [gbs * x for x in xs], ls, color=ROOF_COLOR, lw=1.3)
    if at_top:
        x_top = min(ymax / gbs, ai_max)
        right = x_top > (ai_min * ai_max) ** 0.5           # leaving near the right edge: text to the left
        ax2.annotate(text, (x_top, min(ymax, gbs * ai_max)), xytext=(-4 if right else 4, -4),
                     textcoords="offset points", fontsize=7.5, color=ROOF_COLOR,
                     ha="right" if right else "left", va="top")
    else:
        ax2.annotate(text, (ai_min, gbs * ai_min), xytext=(4, -4), textcoords="offset points",
                     fontsize=7.5, color=ROOF_COLOR, ha="left", va="top")


diagonal(bw_roof, "-", f"DRAM {bw_roof:.0f} GB/s (add)", at_top=True)
diagonal(bw_read, ":", f"DRAM read-only {bw_read:.0f} GB/s", at_top=False)
ax2.axhline(peak, color=ROOF_COLOR, ls="--", lw=1.3)
ax2.annotate(f"NEON FMA peak {peak:.0f} GFLOP/s", (ai_min, peak), xytext=(3, 3), textcoords="offset points",
             fontsize=7.5, color=ROOF_COLOR)
if amx_peak:
    ax2.axhline(amx_peak, color=ROOF_COLOR, ls="-.", lw=1.3)
    ax2.annotate(f"AMX fma32 peak (full tiles) {amx_peak:.0f} GFLOP/s", (ai_min, amx_peak), xytext=(3, 3),
                 textcoords="offset points", fontsize=7.5, color=ROOF_COLOR)
for k in kernels:
    ps = sorted(plateau[k])
    bk.kernel_line(ax2, k, [ai[p] for p in ps], [plateau[k][p] * 2 * mac_per_dof(p) for p in ps])
    if k.startswith("amx"):
        ax2.plot([ai[p] for p in ps], [plateau[k][p] * 2 * mac_per_dof(p) / amx_fill(p) for p in ps],
                 ls=bk.LINESTYLES[k], marker="o", ms=4.5, mfc="none", color=bk.COLORS[k], lw=0.8, alpha=0.7)
    p = ps[-1]                                       # intensity grows with p: name the right end
    ax2.annotate(f"p={p}", (ai[p], plateau[k][p] * 2 * mac_per_dof(p)),
                 fontsize=6.5, color=bk.COLORS.get(k, "C7"), xytext=(4, -2), textcoords="offset points")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(ai_min, ai_max)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel("arithmetic intensity (flop / byte)")
ax2.set_ylabel("GFLOP/s")
ax2.set_title("roofline: sum-factorized flops over compulsory DRAM bytes\n"
              f"(horizontal: compute peaks, diagonal: DRAM; p = {orders[0]} to {orders[-1]} left to right)", fontsize=10.5)
ticks = [t for t in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100) if ai_min <= t <= ai_max]
ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.grid(True, which="major", alpha=0.25)
for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# --- one legend for the kernels below both panels, one column per family:
# baselines | NEON | AMX (matplotlib fills legend columns top to bottom)
def handle(k):
    return Line2D([], [], color=bk.COLORS.get(k, "C7"), ls=bk.LINESTYLES.get(k, "-"), lw=1.6, marker="o", ms=3.5)


columns = [[(handle(k), bk.label(k)) for k in kernels if k in ("serial", "omp", "omp_v")],
           [(handle(k), bk.label(k)) for k in kernels if k.startswith("neon")],
           [(handle(k), bk.label(k)) for k in kernels if k.startswith("amx")]]
if columns[2]:
    columns[2].append((Line2D([], [], color=bk.COLORS["amx_soa"], ls="", marker="o", ms=4.5, mfc="none", alpha=0.7),
                       "AMX raw flops (useful / row fill)"))
columns.append([(handle(k), bk.label(k)) for k in kernels if k not in bk.KERNELS])
columns = [c for c in columns if c]
depth = max(len(c) for c in columns)
blank = (Line2D([], [], ls="", marker=""), "")
entries = [e for c in columns for e in c + [blank] * (depth - len(c))]
bk.figure_legend(fig, [h for h, _ in entries], [l for _, l in entries], ncol=len(columns))

fig.suptitle(title, y=1.0)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(dst, dpi=150, bbox_inches="tight")
print(f"wrote {dst}")
print(f"bandwidth roof {bw_roof:.1f} GB/s (add), read-only {bw_read:.1f} GB/s, compute roof {peak_label}"
      + (f", AMX peak {amx_peak:.0f} GFLOP/s" if amx_peak else ""))
print(f"{'p':>3} {'B/DoF':>7} {'MAC/DoF':>8} {'stB/DoF':>8} {'mem roof':>9} {'FMA roof':>9} {'AMXst roof':>10}  "
      + "  ".join(f"{bk.label(k)[:14]:>14}" for k in kernels))
for p in orders:
    print(f"{p:>3} {bytes_per_dof(p):>7.1f} {mac_per_dof(p):>8.1f} {store_bytes_per_dof(p):>8.1f} "
          f"{mem_roof[p]:>9.2f} {fma_roof[p]:>9.2f} {store_roof[p]:>10.2f}  "
          + "  ".join((f"{plateau[k][p]:>14.2f}" if p in plateau[k] else f"{'-':>14}") for k in kernels))
