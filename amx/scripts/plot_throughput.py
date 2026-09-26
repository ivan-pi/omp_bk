#!/usr/bin/env python3
"""plot_throughput.py -- CEED-style throughput plots from throughput.sh output.

usage: ./plot_throughput.py results.csv [out.png] [title]

One panel per kernel: GDoF/s (linear) versus degrees of freedom (log),
one curve per polynomial order.  A final panel overlays the best order of
each kernel, which is the usual "bake-off" summary view.
"""
import sys
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

src = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".png"
title = sys.argv[3] if len(sys.argv) > 3 else "BK1 (mass operator) throughput"

data = defaultdict(lambda: defaultdict(list))   # data[kernel][p] -> [(dofs, gdofs, target)]
with open(src) as f:
    for row in csv.DictReader(f):
        try:
            g = float(row["gdofs"])
        except ValueError:
            continue
        if math.isnan(g):
            continue
        target = int(row.get("target") or row["dofs"])
        data[row["kernel"]][int(row["p"])].append((int(row["dofs"]), g, target))

labels = {"serial": "serial", "omp": "OpenMP (scalar)", "omp_v": "OpenMP, loops interchanged",
          "neon_aos": "NEON, element-major", "neon_soa": "NEON, elements-on-lanes",
          "amx_aos": "AMX, element-major", "amx_soa": "AMX, elements-on-lanes"}
kernels = [k for k in labels if k in data] + [k for k in data if k not in labels]

# speedup reference: the best NEON kernel, or OpenMP scalar if there is no NEON data
refk = next((k for k in ("neon_soa", "neon_aos", "omp") if k in data), None)
n = len(kernels) + (2 if refk else 1)
cols = min(n, 3)
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.0 * rows), squeeze=False)
axes = axes.flatten()
ymax = max(g for k in data.values() for pts in k.values() for _, g, _ in pts) * 1.08

cmap = plt.get_cmap("viridis")
orders = sorted({p for k in data.values() for p in k})
color = {p: cmap((i + 0.5) / len(orders)) for i, p in enumerate(orders)}


def style(ax, title, ylabel="GDoF/s", ylim=(0, ymax)):
    ax.set_xscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)


for ax, k in zip(axes, kernels):
    for p in sorted(data[k]):
        pts = sorted(data[k][p])
        ax.plot([d for d, _, _ in pts], [g for _, g, _ in pts], "o-", ms=3, lw=1.2,
                color=color[p], label=f"p = {p}")
    style(ax, labels.get(k, k))
    ax.legend(fontsize=8, ncol=2)

# summary panel: for each kernel, its best order at each target size; the
# marker colour says which order won, the line colour which kernel
ax = axes[len(kernels)]
kcolor = {k: f"C{i}" for i, k in enumerate(kernels)}
print(f"{'target':>10}  " + "  ".join(f"{labels.get(k, k):>24}" for k in kernels))
best_all = {}
for k in kernels:
    best = {}
    for p, pts in data[k].items():
        for _, g, t in pts:
            if t not in best or g > best[t][1]:
                best[t] = (p, g)
    best_all[k] = best
    pts = sorted(best.items())
    ax.plot([t for t, _ in pts], [g for _, (_, g) in pts], "-", lw=1.5, color=kcolor[k], label=labels.get(k, k))
    ax.scatter([t for t, _ in pts], [g for _, (_, g) in pts], s=22, zorder=3,
               c=[color[bp] for _, (bp, _) in pts], edgecolors="none")
for t in sorted({t for b in best_all.values() for t in b}):
    print(f"{t:>10}  " + "  ".join(
        f"{('p=%d  %.2f' % best_all[k][t]) if t in best_all[k] else '':>24}" for k in kernels))
style(ax, "best order per kernel (marker colour = order)")
leg1 = ax.legend(fontsize=8, loc="upper left")
ax.add_artist(leg1)
ax.legend([Line2D([], [], marker="o", ls="", color=color[p]) for p in orders],
          [f"p = {p}" for p in orders], fontsize=7, ncol=2, loc="lower right", title="winning order")

# speedup panel: AMX kernels over the reference kernel at equal order
if refk:
    ax = axes[len(kernels) + 1]
    ref = {p: {t: g for _, g, t in pts} for p, pts in data[refk].items()}
    linestyle = {"amx_aos": "--", "amx_soa": "-"}
    for k in ("amx_aos", "amx_soa"):
        if k not in data:
            continue
        for p in sorted(data[k]):
            pts = sorted((t, g / ref[p][t]) for _, g, t in data[k][p] if p in ref and t in ref[p])
            ax.plot([t for t, _ in pts], [s for _, s in pts], linestyle[k], lw=1.2, color=color[p],
                    label=f"p = {p}" if k == "amx_soa" else None)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    style(ax, f"AMX speedup over {labels.get(refk, refk)} at equal order\n"
              "(solid: elements-on-lanes, dashed: element-major)", ylabel="speedup", ylim=None)
    ax.legend(fontsize=7, ncol=2, loc="lower right")

for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle(title, y=1.0)
fig.tight_layout()
fig.savefig(dst, dpi=150, bbox_inches="tight")
print(f"wrote {dst}")
