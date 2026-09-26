"""bkplot.py -- shared pieces of the BK1 plots: the kernels' fixed names, colours
and line styles, the results.csv reader and the plateau rule.

Kernels are a fixed categorical set, so every plot draws the same kernel in the
same colour whatever subset a CSV contains; the two layouts of one kernel share
a hue and differ by line style (solid: elements-on-lanes, dashed: element-major).
"""
import csv
import math
from collections import defaultdict

# fixed drawing order, label and colour (a validated categorical palette;
# in the order of KERNELS)
KERNELS = ["serial", "omp", "omp_v", "neon_aos", "neon_soa", "amx_aos", "amx_soa"]
LABELS = {"serial": "serial", "omp": "OpenMP (scalar)", "omp_v": "OpenMP, loops interchanged",
          "neon_aos": "NEON, element-major", "neon_soa": "NEON, elements-on-lanes",
          "amx_aos": "AMX, element-major", "amx_soa": "AMX, elements-on-lanes"}
COLORS = {"serial": "#4a3aa7", "omp": "#eda100", "omp_v": "#008300",
          "neon_aos": "#e34948", "neon_soa": "#e34948",
          "amx_aos": "#2a78d6", "amx_soa": "#2a78d6"}
LINESTYLES = {k: ("--" if k.endswith("_aos") else "-") for k in KERNELS}

PLATEAU_TOP = 3     # plateau = best of the three largest sizes run for that kernel and order


def label(k):
    return LABELS.get(k, k)


def load_results(path):
    """points[kernel][p] -> [(dofs, gdofs, target)], skipping NaN rows."""
    points = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                g = float(row["gdofs"])
            except ValueError:
                continue
            if math.isnan(g):
                continue
            target = int(row.get("target") or row["dofs"])
            points[row["kernel"]][int(row["p"])].append((int(row["dofs"]), g, target))
    return points


def kernels_in(points):
    """The kernels present, in the fixed drawing order (unknown names last)."""
    return [k for k in KERNELS if k in points] + [k for k in points if k not in KERNELS]


def plateaus(points, top=PLATEAU_TOP):
    """plateau[kernel][p] = best GDoF/s among the `top` largest sizes run."""
    plateau = defaultdict(dict)
    for k, byp in points.items():
        for p, pts in byp.items():
            largest = sorted(pts)[-top:]
            plateau[k][p] = max(g for _, g, _ in largest)
    return plateau


def kernel_line(ax, k, xs, ys, **kw):
    """One kernel's line in its fixed colour and style."""
    args = dict(color=COLORS.get(k, "C7"), ls=LINESTYLES.get(k, "-"), lw=1.6, marker="o", ms=3.5, label=label(k))
    args.update(kw)
    return ax.plot(xs, ys, **args)


def figure_legend(fig, handles, labels, ncol, y=0.0):
    """One legend for the whole figure, below the panels, so no panel is covered."""
    return fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
                      frameon=False, fontsize=9, handlelength=2.6)
