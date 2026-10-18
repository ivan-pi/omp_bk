#!/usr/bin/env python3
"""plot_epg.py -- bar charts of the batch kernel's plateau against the elements
per gang, one panel per polynomial order, from a throughput.sh sweep whose
KERNELS tokens are batch@<path>_epg<N> builds (make -C acc BK1_acc -B
DEFS=-DBK_EPG=N) plus the default `batch`.

usage: acc/scripts/plot_epg.py epg.csv [out.png] [figure title]

Per order the plateau is the best of the two largest sizes run. BK_EPG is
clamped to the 48 KB shared-memory limit inside the kernel, so builds that
clamp to the same batch at an order are one configuration: their plateaus
are averaged and the spread is printed as the run-to-run noise. The best bar
is drawn darker and labeled; the table default of acc/BK1_acc.cpp (TABLE
below, keep it in step with batch_elements) is marked under its tick. Rows
of the plain `batch` kernel are ignored: the CSV does not say which table
they were measured with. Needs matplotlib.
"""
import csv
import re
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "epg.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".png"
title = sys.argv[3] if len(sys.argv) > 3 else None

REFERENCE = {1: 65, 2: 82, 3: 88, 4: 85, 5: 80, 6: 78, 7: 72, 8: 70}   # dealii-X batched kernel, GH200, fp32


def cap(nq):
    """Elements per gang that fit the 48 KB static shared memory (two nq^3 boxes of floats)."""
    return (48 * 1024) // (2 * nq * nq * nq * 4)


TABLE = {3: 10, 4: 6, 5: 5, 6: 3, 7: 3, 8: 2, 9: 3, 10: 6}   # batch_elements of acc/BK1_acc.cpp


def table_default(nq):
    """The table of acc/BK1_acc.cpp, for the marker under the default's tick."""
    return TABLE.get(nq)


# plateau[(p, kernel)] = best of the two largest sizes
rows = defaultdict(list)
for r in csv.DictReader(open(src)):
    try:
        g = float(r["gdofs"])
    except ValueError:
        continue
    rows[(int(r["p"]), r["kernel"])].append((int(r["dofs"]), g))
plateau = {k: max(g for _, g in sorted(v)[-2:]) for k, v in rows.items()}

# group by effective elements per gang
per_order = defaultdict(lambda: defaultdict(list))   # p -> epg -> [plateaus]
for (p, kernel), g in plateau.items():
    nq = p + 2
    m = re.search(r"_epg(\d+)$", kernel)
    if not m:
        continue
    epg = min(int(m.group(1)), cap(nq))
    per_order[p][epg].append(g)

orders = sorted(per_order)
fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), squeeze=False)
ymax = max(max(max(v) for v in d.values()) for d in per_order.values())
ymax = max(ymax, max(REFERENCE.values())) * 1.12
BLUE, DARK, INK, MUTED, GRID = "#2a78d6", "#174a86", "#0b0b0b", "#52514e", "#dddcd8"

print(f"{'p':>2} {'nq':>3}  best epg  GDoF/s   noise (configs measured more than once)")
for ax, p in zip(axes.flatten(), orders):
    nq = p + 2
    epgs = sorted(per_order[p])
    means = [statistics.fmean(per_order[p][e]) for e in epgs]
    best = max(range(len(epgs)), key=lambda i: means[i])
    dup = [per_order[p][e] for e in epgs if len(per_order[p][e]) > 1]
    noise = ""
    if dup:
        rel = max((max(v) - min(v)) / statistics.fmean(v) for v in dup)
        noise = f"{100 * rel:.1f} percent spread over {max(len(v) for v in dup)} repeats"
    print(f"{p:>2} {nq:>3}  {epgs[best]:>8}  {means[best]:6.1f}   {noise}")

    x = range(len(epgs))
    ax.bar(x, means, width=0.72, color=[DARK if i == best else BLUE for i in x], linewidth=0)
    ax.text(best, means[best] + ymax * 0.015, f"{means[best]:.0f}", ha="center", va="bottom",
            fontsize=8, color=INK)
    ax.axhline(REFERENCE[p], color=MUTED, lw=1, zorder=0)
    ax.text(len(epgs) - 0.5, REFERENCE[p] + ymax * 0.01, f"dealii-X {REFERENCE[p]}",
            ha="right", va="bottom", fontsize=7, color=MUTED)
    d = table_default(nq)
    if d in epgs:
        ax.annotate("\u25b2", xy=(epgs.index(d), 0), xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color=INK, annotation_clip=False)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(e) for e in epgs], fontsize=7)
    ax.set_ylim(0, ymax)
    ax.set_title(f"p = {p}, nq = {nq}", fontsize=10, color=INK)
    ax.set_xlabel("elements per gang", fontsize=8, color=MUTED, labelpad=10)
    ax.set_ylabel("GDoF/s", fontsize=8, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
for ax in axes.flatten()[len(orders):]:
    ax.set_visible(False)
if title:
    fig.suptitle(title, color=INK)
fig.text(0.01, 0.005, "dark bar: best measured; triangle: the table default of BK1_acc.cpp; "
         "gray line: the dealii-X batched kernel on GH200 (fp32)", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(dst, dpi=150)
print(f"wrote {dst}")
