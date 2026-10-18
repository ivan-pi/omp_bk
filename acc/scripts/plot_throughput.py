#!/usr/bin/env python3
"""plot_throughput.py -- CEED-style throughput plots from throughput.sh output.

usage: acc/scripts/plot_throughput.py results.csv [out.png] [figure title]

One panel per executable in the CSV (in order of first appearance): GDoF/s
(linear, shared axis) versus degrees of freedom (log), one curve per
polynomial order with its own legend entry.  Needs matplotlib.
"""
import csv
import math
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".png"
title = sys.argv[3] if len(sys.argv) > 3 else None


def load_results(path):
    """points[kernel][p] -> [(dofs, gdofs)], kernels in order of first appearance."""
    points = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                g = float(row["gdofs"])
            except ValueError:
                continue
            if math.isnan(g):
                continue
            byp = points.setdefault(row["kernel"], defaultdict(list))
            byp[int(row["p"])].append((int(row["dofs"]), g))
    return points


data = load_results(src)
if not data:
    sys.exit(f"no data points in {src}")
kernels = list(data)
orders = sorted({p for byp in data.values() for p in byp})
ymax = max(g for byp in data.values() for pts in byp.values() for _, g in pts) * 1.08
colors = plt.get_cmap("tab10")
ocolor = {p: colors(i % 10) for i, p in enumerate(orders)}

fig, axes = plt.subplots(1, len(kernels), figsize=(5.2 * len(kernels), 3.9), squeeze=False)
for ax, k in zip(axes[0], kernels):
    for p in sorted(data[k]):
        pts = sorted(data[k][p])
        ax.plot([d for d, _ in pts], [g for _, g in pts], "o-", ms=2.5, lw=1.2,
                color=ocolor[p], label=f"p={p}")
    ax.set_xscale("log")
    ax.set_ylim(0, ymax)
    ax.set_title(k, fontsize=11)
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("GDoF/s")
    ax.grid(True, which="major", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
if title:
    fig.suptitle(title)
fig.tight_layout()
fig.savefig(dst, dpi=150)
print(f"wrote {dst}")
