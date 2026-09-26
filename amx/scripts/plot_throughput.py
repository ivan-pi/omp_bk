#!/usr/bin/env python3
"""plot_throughput.py -- CEED-style throughput plots from throughput.sh output.

usage: ./plot_throughput.py results.csv [out.png] [title]

One panel per kernel: GDoF/s (linear) versus degrees of freedom (log), one
curve per polynomial order (a light-to-dark ramp, legend once for the whole
figure), and a last panel with the AMX speedup over NEON at the plateau (best
of the largest sizes) as a function of the order.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import bkplot as bk

src = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".png"
title = sys.argv[3] if len(sys.argv) > 3 else "BK1 (mass operator, fp32) throughput"

data = bk.load_results(src)
kernels = bk.kernels_in(data)
plateau = bk.plateaus(data)
orders = sorted({p for k in data.values() for p in k})
ymax = max(g for k in data.values() for pts in k.values() for _, g, _ in pts) * 1.08

# one hue, light to dark with the order (an ordinal ramp)
cmap = plt.get_cmap("viridis")
ocolor = {p: cmap(0.9 - 0.85 * i / max(1, len(orders) - 1)) for i, p in enumerate(orders)}

n = len(kernels) + 1
cols = 3
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.9 * rows), squeeze=False)
axes = axes.flatten()


def style(ax, title, ylabel="GDoF/s", ylim=(0, ymax), xlog=True):
    if xlog:
        ax.set_xscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("degrees of freedom" if xlog else "polynomial order p")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --- one panel per kernel, one curve per order ------------------------------
for ax, k in zip(axes, kernels):
    for p in sorted(data[k]):
        pts = sorted(data[k][p])
        ax.plot([d for d, _, _ in pts], [g for _, g, _ in pts], "o-", ms=2.5, lw=1.2, color=ocolor[p])
    style(ax, bk.label(k))

# --- AMX speedup over NEON at the plateau, per order ------------------------
ax = axes[len(kernels)]
drawn = False
for amx, neon in (("amx_soa", "neon_soa"), ("amx_aos", "neon_aos")):
    if amx in plateau and neon in plateau:
        ps = sorted(set(plateau[amx]) & set(plateau[neon]))
        bk.kernel_line(ax, amx, ps, [plateau[amx][p] / plateau[neon][p] for p in ps],
                       label=("elements-on-lanes" if amx.endswith("_soa") else "element-major") + " (AMX / NEON)")
        drawn = True
if drawn:
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(orders)
    style(ax, f"AMX over NEON at the plateau (best of the {bk.PLATEAU_TOP} largest sizes)",
          ylabel="speedup", ylim=None, xlog=False)
    ax.legend(fontsize=8, frameon=False, loc="best")
else:
    ax.set_visible(False)
for ax in axes[n + 1:]:
    ax.set_visible(False)

# --- the order legend, once: vertical in a spare panel, else below the figure -
order_handles = [Line2D([], [], marker="o", ls="", color=ocolor[p]) for p in orders]
order_labels = [f"p = {p}" for p in orders]
if n < len(axes):
    spare = axes[n]
    spare.set_visible(True)
    spare.axis("off")
    spare.legend(order_handles, order_labels, loc="center", ncol=2, frameon=False, fontsize=10,
                 title="polynomial order", title_fontsize=11, columnspacing=2.5, labelspacing=0.9)
else:
    bk.figure_legend(fig, order_handles, order_labels, ncol=min(14, len(orders)))

# plateau table on stdout: GDoF/s per kernel and order
print(f"{'p':>3}  " + "  ".join(f"{bk.label(k)[:14]:>14}" for k in kernels))
for p in orders:
    print(f"{p:>3}  " + "  ".join((f"{plateau[k][p]:>14.2f}" if p in plateau[k] else f"{'-':>14}") for k in kernels))

fig.suptitle(title, y=1.0)
fig.tight_layout(rect=(0, 0.04 if n >= len(axes) else 0, 1, 1))
fig.savefig(dst, dpi=150, bbox_inches="tight")
print(f"wrote {dst}")
