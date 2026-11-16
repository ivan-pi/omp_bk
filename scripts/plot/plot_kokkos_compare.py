#!/usr/bin/env python3
"""plot_kokkos_compare.py -- our BK1, BK3, BK5 kernels against the Kokkos
CEED_BK kernels (dealii-X/benchmarks) at a saturating size.

usage:  scripts/plot/plot_kokkos_compare.py kokkos_compare.csv out.png [subtitle]

One panel per BK, grouped bars over the polynomial degree p: our fastest
OpenMP offload variant, our fastest SYCL variant and Kokkos with its fastest
element batch (median GDoF/s over the repetitions in the CSV of
scripts/study/kokkos_compare.sh).  A dark tick on the Kokkos bar marks its
default batch; the label above each group is the speed-up of our faster
kernel over the fastest Kokkos run.  Colours: the first three categorical
slots (validated), in fixed order.
"""

import csv
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
IMPLS = (("omp", "ours: OpenMP offload", "#2a78d6"),
         ("sycl", "ours: SYCL", "#eb6834"),
         ("kokkos", "Kokkos CEED_BK (best batch)", "#1baf7a"))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, out = sys.argv[1], sys.argv[2]
    subtitle = sys.argv[3] if len(sys.argv) > 3 else ""
    runs = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r["check"] != "FAIL":
            runs[(int(r["bk"]), int(r["p"]), r["impl"], r["variant"])].append(float(r["gdofs"]))
    best = {}      # (bk, p, impl) -> (variant, median GDoF/s)
    default = {}   # (bk, p) -> Kokkos default batch, median GDoF/s
    for (bk, p, impl, v), x in runs.items():
        m = statistics.median(x)
        if best.get((bk, p, impl), ("", 0.0))[1] < m:
            best[(bk, p, impl)] = (v, m)
        if impl == "kokkos" and v.startswith("d"):
            default[(bk, p)] = m
    bks = sorted({k[0] for k in best})

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2})
    fig, axes = plt.subplots(1, len(bks), figsize=(5.2 * len(bks), 4.8), facecolor=SURFACE,
                             squeeze=False)
    width = 0.26
    for ax, bk in zip(axes[0], bks):
        ax.set_facecolor(SURFACE)
        ps = sorted({k[1] for k in best if k[0] == bk})
        ymax = 0.0
        for i, (impl, label, color) in enumerate(IMPLS):
            xs = [p + (i - 1) * width for p in ps if (bk, p, impl) in best]
            ys = [best[(bk, p, impl)][1] for p in ps if (bk, p, impl) in best]
            ax.bar(xs, ys, width, color=color, edgecolor=SURFACE, linewidth=1.0, label=label)
            ymax = max([ymax] + ys)
        for p in ps:
            if (bk, p) in default:
                ax.plot([p + width / 2 + 0.03, p + 3 * width / 2 - 0.03], [default[(bk, p)]] * 2,
                        color=TEXT, linewidth=1.6, solid_capstyle="butt")
            ours = max(best.get((bk, p, i), ("", 0.0))[1] for i in ("omp", "sycl"))
            kok = best.get((bk, p, "kokkos"), ("", 0.0))[1]
            top = max(ours, kok)
            if kok > 0:
                ax.annotate(f"×{ours / kok:.2f}", xy=(p, top), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8, color=TEXT2)
        ax.set_xticks(ps)
        ax.set_xlabel("polynomial degree p")
        ax.set_ylabel("GDoF/s")
        ax.set_ylim(0, ymax * 1.12)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"BK{bk}", loc="left", color=TEXT, fontsize=11)
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=TEXT, linewidth=1.6))
    labels.append("Kokkos, default batch")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=TEXT2, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.005, "Arc A770, icpx 2026.1, float; median of runs at a saturating size; "
             "labels: our faster kernel over the fastest Kokkos batch" + (f"; {subtitle}" if subtitle else ""),
             ha="center", va="bottom", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
