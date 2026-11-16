#!/usr/bin/env python3
"""plot_compare.py -- BK1, BK3, BK5 throughput of several implementations at a
saturating size: our OpenMP offload and SYCL kernels against Kokkos
(dealii-X CEED_BK, scripts/study/kokkos_compare.sh) or tinytc
(scripts/study/bk3_tinytc_compare.sh; best configuration per order, in tinytc's
blocked layout and in ours).

usage:  scripts/plot/plot_compare.py compare.csv out.png [ref] [subtitle] [missing]
        CSV: bk,p,impl,variant,rep,gdofs,check   (impl: omp, sycl, kokkos, tinytc,
        tinytc_ec)

One panel per BK, grouped bars over the polynomial degree p: the fastest
variant of each implementation in the CSV (median GDoF/s over the
repetitions).  The label above each group is the speed-up of our faster
kernel (omp, sycl) over ref (default kokkos), or, with ref = tinytc, of
tinytc over our faster kernel.  Kokkos: a dark tick marks its default batch
(variant db<n>).  Colours: one categorical slot per implementation, in fixed
order (validated).
"""

import csv
import statistics
import sys
import textwrap
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
         ("kokkos", "Kokkos CEED_BK (best batch)", "#1baf7a"),
         ("tinytc", "tinytc, blocked layout", "#eda100"),
         ("tinytc_ec", "tinytc, our layout", "#e87ba4"))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, out = sys.argv[1], sys.argv[2]
    ref = sys.argv[3] if len(sys.argv) > 3 else "kokkos"
    subtitle = sys.argv[4] if len(sys.argv) > 4 else ""
    missing = sys.argv[5] if len(sys.argv) > 5 else ""   # label where ref has no result
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
    impls = [i for i in IMPLS if any(k[2] == i[0] for k in best)]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2})
    fig, axes = plt.subplots(1, len(bks), figsize=(max(5.2 * len(bks), 9.0), 4.8), facecolor=SURFACE,
                             squeeze=False)
    width = 0.78 / len(impls)
    for ax, bk in zip(axes[0], bks):
        ax.set_facecolor(SURFACE)
        ps = sorted({k[1] for k in best if k[0] == bk})
        ymax = 0.0
        for i, (impl, label, color) in enumerate(impls):
            xs = [p + (i - (len(impls) - 1) / 2) * width for p in ps if (bk, p, impl) in best]
            ys = [best[(bk, p, impl)][1] for p in ps if (bk, p, impl) in best]
            ax.bar(xs, ys, width, color=color, edgecolor=SURFACE, linewidth=1.0, label=label)
            ymax = max([ymax] + ys)
        names = [i[0] for i in impls]
        for p in ps:
            if (bk, p) in default and "kokkos" in names:
                x = p + (names.index("kokkos") - (len(impls) - 1) / 2) * width
                ax.plot([x - width / 2 + 0.03, x + width / 2 - 0.03], [default[(bk, p)]] * 2,
                        color=TEXT, linewidth=1.6, solid_capstyle="butt")
            ours = max(best.get((bk, p, i), ("", 0.0))[1] for i in ("omp", "sycl"))
            other = best.get((bk, p, ref), ("", 0.0))[1]
            top = max([ours, other] + [best.get((bk, p, i), ("", 0.0))[1] for i in names])
            if other > 0 and ours > 0:
                ratio = other / ours if ref == "tinytc" else ours / other
                ax.annotate(f"×{ratio:.2f}", xy=(p, top), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8, color=TEXT2)
            elif ours > 0 and missing:
                ax.annotate(missing, xy=(p, top), xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=7.5, color=TEXT2)
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
    if default:
        handles.append(plt.Line2D([], [], color=TEXT, linewidth=1.6))
        labels.append("Kokkos, default batch")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=TEXT2, bbox_to_anchor=(0.5, 0.035))
    note = ("Arc A770, icpx 2026.1, float; median of runs at a saturating size; "
            + ("labels: tinytc (blocked) over our faster kernel" if ref == "tinytc" else
               "labels: our faster kernel over " + ("the fastest Kokkos batch" if ref == "kokkos" else ref))
            + (f"; {subtitle}" if subtitle else ""))
    fig.text(0.5, 0.005, textwrap.fill(note, int(16 * fig.get_figwidth())), ha="center", va="bottom",
             fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
