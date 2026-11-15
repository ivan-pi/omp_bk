#!/usr/bin/env python3
"""plot_bk5_study.py -- BK5 symmetric half (G6) vs full (G9) geometric tensor.

usage:  scripts/plot/plot_bk5_study.py bk5_study.csv out.png [stream_copy_GBs]

Two panels over the polynomial degree p: GDoF/s (left) and GB/s (right, with
the STREAM copy bandwidth as a dashed line).  Colour = tensor storage (G6
blue, G9 orange: the first two categorical slots, validated); line style =
kernel (per-point kernel, best batch and layout per order: solid with
circles; the original one-element-per-thread kernel: dotted with squares).
Values are medians over the repetitions in the CSV.  The left panel also
shows the bandwidth-model ratio G6/G9 = 44/32 = 1.375 at each p as text.
"""

import csv
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
COLOR = {6: "#2a78d6", 9: "#eb6834"}


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, out = sys.argv[1], sys.argv[2]
    stream = float(sys.argv[3]) if len(sys.argv) > 3 else 459.0
    g = defaultdict(list)
    b = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["relerr"] == "FAIL":
            continue
        k = (int(r["p"]), r["variant"])
        g[k].append(float(r["gdofs"]))
        b[k].append(float(r["gbs"]))
    ps = sorted({k[0] for k in g})
    series = {}   # (F, kernel) -> [(p, gdofs, gbs, variant)]
    for F in (6, 9):
        for kern, rx in (("pt", rf"^pt{F}"), ("ref", rf"^ref{F}")):
            pts = []
            for p in ps:
                ks = [k for k in g if k[0] == p and re.search(rx, k[1])]
                if not ks:
                    continue
                best = max(ks, key=lambda k: statistics.median(g[k]))
                pts.append((p, statistics.median(g[best]), statistics.median(b[best]), best[1]))
            series[(F, kern)] = pts

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=SURFACE)
    for c, (ax, ylabel, title) in enumerate(zip(axes, ("GDoF/s", "GB/s"),
                                                ("BK5 throughput", "Effective bandwidth (in + out + G)"))):
        ax.set_facecolor(SURFACE)
        ymax = 0.0
        for (F, kern), pts in series.items():
            if not pts:
                continue
            xs = [v[0] for v in pts]
            ys = [v[1 + c] for v in pts]
            ymax = max(ymax, max(ys))
            name = ("per-point kernel" if kern == "pt" else "original kernel") + f", G{F}"
            ax.plot(xs, ys, color=COLOR[F], linewidth=2, linestyle="-" if kern == "pt" else ":",
                    marker="o" if kern == "pt" else "s", markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.0, label=name)
        if c == 1:
            ax.axhline(stream, color=TEXT2, linewidth=1.2, linestyle=(0, (5, 3)))
            ax.annotate(f"STREAM copy {stream:.0f} GB/s", xy=(ps[0], stream), xytext=(2, 5),
                        textcoords="offset points", fontsize=8.5, color=TEXT2)
            ymax = max(ymax, stream)
        else:
            g6 = {v[0]: v[1] for v in series[(6, "pt")]}
            g9 = {v[0]: v[1] for v in series[(9, "pt")]}
            for p in ps:
                if p in g6 and p in g9:
                    ax.annotate(f"×{g6[p] / g9[p]:.2f}", xy=(p, g6[p]), xytext=(0, 8),
                                textcoords="offset points", ha="center", fontsize=8, color=TEXT2)
        ax.set_xticks(ps)
        ax.set_xlabel("polynomial degree p (nq = p + 1 points per direction)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ymax * 1.15)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=TEXT2, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.005, "BK5 on Arc A770, OpenMP, icpx 2026.1, float, 5e7 DoFs, median of runs; "
             "labels: G6/G9 speed ratio of the per-point kernel (bandwidth model: 1.375)",
             ha="center", va="bottom", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
