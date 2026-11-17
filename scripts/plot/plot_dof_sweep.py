#!/usr/bin/env python3
"""plot_dof_sweep.py -- GDoF/s and GB/s versus DoFs from dof_sweep.sh.

usage:  scripts/plot/plot_dof_sweep.py results.csv out.png [--device] [--p-offset=N] [--title=TEXT]
                                      [--apis=a,b,...] [--kernel-labels] [--ntests=N]

The rows are the values of the csv's api column (OpenMP/SYCL, or any other
labels, e.g. G6/G9 from bk5_sweep.sh); p = nq - N (default N = 2, BK1;
N = 1 for BK5, collocated); --title replaces the footnote's first part;
--apis keeps those api values, as rows in that order (e.g. omp6,sycl6 from
bk5_sweep.sh); --kernel-labels adds each line's kernel (the csv's kernel
column) to its end label; --ntests gives the launches per point of the
footnote (default 20).

A 2 x 2 grid: rows = API (OpenMP top, SYCL bottom), columns = GDoF/s (left)
and GB/s (right, with the STREAM copy bandwidth from results.csv.stream as a
dashed line).  Each column shares its y-axis so the two APIs compare
directly.  Colour = polynomial degree p, in the fixed order of the reference
categorical palette (validated for lines on the adjacent pairlist; slots
below 3:1 contrast are relieved by the direct p labels at the line ends and
by the CSV as the table view).  --device plots SYCL's device (event) times
instead of host times (any row with device times, e.g. sycl6/sycl9).
"""

import csv
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
# categorical slots 1..8 in their fixed order; p = 1 takes slot 1
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
API_NAME = {"omp": "OpenMP", "sycl": "SYCL", "G6": "G6 (symmetric half)", "G9": "G9 (full tensor)",
            "omp6": "OpenMP, G6", "sycl6": "SYCL, G6", "omp9": "OpenMP, G9", "sycl9": "SYCL, G9"}


def load(path, device, p_offset=2, kernels=None):
    series = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if kernels is not None:
                kernels[(row["api"], int(row["nq"]) - p_offset)] = row["kernel"]
            g, b = row["gdofs"], row["gbs"]
            if device and row["gdofs_dev"]:
                g, b = row["gdofs_dev"], row["gbs_dev"]
            if g in ("", "NaN"):
                continue
            key = (row["api"], int(row["nq"]) - p_offset)
            series.setdefault(key, {})[int(row["dofs"])] = (float(g), float(b))
    return {k: sorted((d, v[0], v[1]) for d, v in pts.items()) for k, pts in series.items()}


def stream_copy(path):
    try:
        for line in open(path + ".stream"):
            m = re.match(r"copy\s.*GB/s\s*=\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
    except OSError:
        pass
    return None


def end_labels(ax, ends, ymin_gap, ymax_label=None):
    """Label each line at its right end, nudging labels apart vertically and,
    with ymax_label, keeping the stack below that value (e.g. a reference line)."""
    ends = sorted(ends, key=lambda e: e[1])
    placed = []
    for x, y, text in ends:
        ylab = y
        if placed and ylab - placed[-1][2] < ymin_gap:
            ylab = placed[-1][2] + ymin_gap
        placed.append([x, y, ylab, text])
    if ymax_label is not None and placed and placed[-1][2] > ymax_label:
        shift = placed[-1][2] - ymax_label
        for lab in placed:
            lab[2] -= shift
    # a label moved off its line end gets a thin leader line back to it
    x_text = ax.get_xlim()[1] if ax.get_xscale() != "log" else None
    for x, y, ylab, text in placed:
        moved = abs(ylab - y) > 0.25 * ymin_gap
        ax.annotate(text, xy=(x, y), xytext=(x * 1.55 if x_text is None else x, ylab),
                    textcoords="data", va="center", ha="left", fontsize=8.5, color=TEXT2,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8, shrinkA=0, shrinkB=3)
                    if moved else None)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, out = sys.argv[1], sys.argv[2]
    device = "--device" in sys.argv[3:]
    p_offset = 2
    title = "BK1 on Arc A770, icpx 2026.1, float"
    keep = None
    ntests = 20
    for a in sys.argv[3:]:
        if a.startswith("--p-offset="):
            p_offset = int(a.split("=", 1)[1])
        if a.startswith("--title="):
            title = a.split("=", 1)[1]
        if a.startswith("--ntests="):
            ntests = int(a.split("=", 1)[1])
        if a.startswith("--apis="):
            keep = a.split("=", 1)[1].split(",")
    kernel_labels = "--kernel-labels" in sys.argv[3:]
    kernels = {}
    series = load(path, device, p_offset, kernels)
    if keep is not None:
        series = {k: v for k, v in series.items() if k[0] in keep}
    copy_bw = stream_copy(path)
    degrees = sorted({p for (_, p) in series})
    apis = sorted({k[0] for k in series}, key=lambda a: (a not in ("omp", "G6"), a))
    if keep is not None:
        apis = [a for a in keep if a in apis]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2})
    fig, axes = plt.subplots(len(apis), 2, figsize=(12.5, 4.3 * len(apis) + 0.9), facecolor=SURFACE,
                             squeeze=False)
    col_max = [0.0, copy_bw or 0.0]
    for pts in series.values():
        col_max[0] = max(col_max[0], max(v[1] for v in pts))
        col_max[1] = max(col_max[1], max(v[2] for v in pts))
    for r, api in enumerate(apis):
        for c, ylabel in enumerate(("GDoF/s", "GB/s")):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ends = []
            for p in degrees:
                if (api, p) not in series:
                    continue
                pts = series[(api, p)]
                xs = [v[0] for v in pts]
                ys = [v[1 + c] for v in pts]
                ax.plot(xs, ys, color=SLOTS[(p - 1) % len(SLOTS)], linewidth=2, marker="o", markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0, label=f"p = {p} (nq = {p + p_offset})")
                label = f"p = {p}, {kernels[(api, p)]}" if kernel_labels else f"p = {p}"
                ends.append((xs[-1], ys[-1], label))
            ymax = col_max[c]
            cap = None
            if c == 1 and copy_bw:
                ax.axhline(copy_bw, color=TEXT2, linewidth=1.2, linestyle=(0, (5, 3)))
                ax.annotate(f"STREAM copy {copy_bw:.0f} GB/s", xy=(1e3, copy_bw), xytext=(2, 5),
                            textcoords="offset points", fontsize=8.5, color=TEXT2)
                cap = copy_bw - ymax * 0.05
            ax.set_xscale("log")
            ax.set_xlim(8e2, 3.5e8)
            ax.set_ylim(0, ymax * 1.12)
            ax.set_ylabel(ylabel)
            if r == len(apis) - 1:
                ax.set_xlabel("DoFs (elements × (p+1)³)")
            ax.grid(True, which="major", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            timing = "device time" if (device and api.startswith("sycl")) else "host time"
            what = "throughput" if c == 0 else ("effective bandwidth (in + out + JxW)" if p_offset == 2 else "effective bandwidth (in + out + G)")
            ax.set_title(f"{API_NAME.get(api, api)} — {what}, {timing}", loc="left", color=TEXT, fontsize=10.5)
            end_labels(ax, ends, ymin_gap=ymax * 0.045, ymax_label=cap)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(degrees), frameon=False, fontsize=8.5,
               labelcolor=TEXT2, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.005, f"{title}, best of {ntests} launches", ha="center",
             va="bottom", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
