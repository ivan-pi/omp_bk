#!/usr/bin/env python3
"""plot_roofline.py -- roofline of the BK1 and BK5 kernels on one GPU.

usage:  scripts/plot/plot_roofline.py bk1_sweep.csv bk5_sweep.csv out.png
            [--peak=GFLOPs] [--bw=GBs] [--device]

Top row, three panels (BK1, BK5 G6, BK5 G9), log-log: performance in GFLOP/s over
arithmetic intensity in flop/byte, with the memory roof (STREAM copy, read
from bk5_sweep.csv.stream unless --bw is given) and the FP32 FMA peak
(default: Arc A770, 512 XVEs x 8 lanes x 2 flops x 2.4 GHz = 19661 GFLOP/s).
One point per polynomial degree p and API, at the largest problem size of
the sweep (host times; --device uses SYCL's device times).  Colour = p (the
fixed categorical order of plot_dof_sweep.py), marker = API (OpenMP circle,
SYCL diamond, hollow, under the circle).  Bottom row: the share of the
roof each point reaches (the same in GFLOP/s and GDoF/s) over p.  A table of
the points goes to stdout.

Work and traffic per DoF (a DoF is a mode for BK1, a point for BK5; the
algorithmic flops of the sum-factorised operator, an FMA counting 2):
  BK1, nm = p + 1 modes, nq = p + 2 points per direction:
      flops = [4 (nq nm^3 + nq^2 nm^2 + nq^3 nm) + nq^3] / nm^3
      bytes = 4 (2 + (nq / nm)^3)            in + out + JxW
  BK5, nq = p + 1:
      flops = 12 nq + 20                     3 derivatives, G (9 FMA), 3
                                             transposed, the final sum
      bytes = 4 (2 + F), F = 6 (G6) or 9 (G9)
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
ROOF = "#52514e"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def bk1_cost(p):
    nm, nq = p + 1, p + 2
    flops = (4 * (nq * nm ** 3 + nq ** 2 * nm ** 2 + nq ** 3 * nm) + nq ** 3) / nm ** 3
    return flops, 4 * (2 + (nq / nm) ** 3)


def bk5_cost(p, F):
    return 12 * (p + 1) + 20, 4 * (2 + F)


def largest(path, device):
    """(api, nq) -> GDoF/s at the largest DoF count of the sweep"""
    best = {}
    for r in csv.DictReader(open(path)):
        g = r["gdofs_dev"] if (device and r.get("gdofs_dev")) else r["gdofs"]
        if g in ("", "NaN"):
            continue
        key = (r["api"], int(r["nq"]))
        d = int(r["dofs"])
        if key not in best or d > best[key][0]:
            best[key] = (d, float(g))
    return {k: v[1] for k, v in best.items()}


def stream_copy(path):
    try:
        for line in open(path + ".stream"):
            m = re.match(r"copy\s.*GB/s\s*=\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
    except OSError:
        pass
    return None


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    bk1_path, bk5_path, out = sys.argv[1:4]
    peak = 512 * 8 * 2 * 2.4
    bw = None
    device = "--device" in sys.argv[4:]
    for a in sys.argv[4:]:
        if a.startswith("--peak="):
            peak = float(a.split("=", 1)[1])
        if a.startswith("--bw="):
            bw = float(a.split("=", 1)[1])
    bw = bw or stream_copy(bk5_path) or stream_copy(bk1_path) or 459.0
    ridge = peak / bw

    bk1 = largest(bk1_path, device)
    bk5 = largest(bk5_path, device)
    # panel -> list of (p, api, intensity, GFLOP/s, GDoF/s, roof GDoF/s)
    panels = {"BK1 (mass)": [], "BK5 G6 (symmetric half)": [], "BK5 G9 (full tensor)": []}
    for (api, nq), g in bk1.items():
        p = nq - 2
        fl, by = bk1_cost(p)
        panels["BK1 (mass)"].append((p, api, fl / by, g * fl, g, min(bw / by, peak / fl)))
    for (api, nq), g in bk5.items():
        p = nq - 1
        F = int(api[-1])
        fl, by = bk5_cost(p, F)
        name = "BK5 G6 (symmetric half)" if F == 6 else "BK5 G9 (full tensor)"
        panels[name].append((p, api.rstrip("69"), fl / by, g * fl, g, min(bw / by, peak / fl)))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT2})
    fig, grid = plt.subplots(2, 3, figsize=(16.5, 9.4), facecolor=SURFACE, sharey="row",
                             gridspec_kw={"height_ratios": [1.6, 1]})
    axes = grid[0]
    xlo, xhi = 0.5, 200.0
    ylo, yhi = 100.0, 40000.0
    print(f"roofs: memory {bw:.0f} GB/s, FP32 peak {peak:.0f} GFLOP/s, ridge {ridge:.1f} flop/B")
    print(f"{'panel':26s} {'api':5s} {'p':>2s} {'flop/B':>7s} {'GFLOP/s':>8s} {'GDoF/s':>7s} "
          f"{'roof GDoF/s':>11s} {'of roof':>7s} {'of peak':>7s}")
    for ax, (title, pts) in zip(axes, panels.items()):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        # roofs
        ax.plot([xlo, ridge], [bw * xlo, peak], color=ROOF, linewidth=1.4, linestyle=(0, (5, 3)))
        ax.plot([ridge, xhi], [peak, peak], color=ROOF, linewidth=1.4)
        ax.annotate(f"FP32 FMA peak {peak / 1000:.1f} TFLOP/s", xy=(xhi, peak), xytext=(-4, 5),
                    textcoords="offset points", ha="right", fontsize=8.5, color=TEXT2)
        ax.annotate(f"STREAM copy {bw:.0f} GB/s", xy=(14.0, bw * 14.0), xytext=(-6, 8),
                    textcoords="offset points", rotation=31, ha="right", fontsize=8.5, color=TEXT2)
        ax.axvline(ridge, color=GRID, linewidth=0.8, zorder=0)
        ax.annotate(f"ridge {ridge:.0f} flop/B", xy=(ridge, ylo), xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color=TEXT2)
        top = {}
        for p, api, ai, gf, gd, roof in sorted(pts):
            col = SLOTS[(p - 1) % len(SLOTS)]
            if api == "omp":
                ax.plot(ai, gf, marker="o", markersize=6, color=col, markeredgecolor=SURFACE,
                        markeredgewidth=1.0, linestyle="none", zorder=4)
            else:
                ax.plot(ai, gf, marker="D", markersize=10, markerfacecolor=SURFACE, markeredgecolor=col,
                        markeredgewidth=1.6, linestyle="none", zorder=3)
            top[p] = max(top.get(p, (0, 0)), (gf, ai))
            print(f"{title:26s} {api:5s} {p:2d} {ai:7.2f} {gf:8.0f} {gd:7.2f} {roof:11.2f} "
                  f"{100 * gd / roof:6.0f}% {100 * gf / peak:6.1f}%")
        # p labels: odd p above, even p below the roof line, clear of the markers
        for p, (gf, ai) in top.items():
            up = p % 2 == 1
            ax.annotate(f"{p}", xy=(ai, gf), xytext=(-9 if up else 9, 9 if up else -14), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT2)
        # bottom: share of the roof over p
        eff = grid[1][list(panels).index(title)]
        eff.set_facecolor(SURFACE)
        for api, marker in (("sycl", "D"), ("omp", "o")):
            ser = sorted((p, 100 * gd / roof) for p, a, ai, gf, gd, roof in pts if a == api)
            if not ser:
                continue
            xs = [v[0] for v in ser]
            ys = [v[1] for v in ser]
            eff.plot(xs, ys, color=TEXT2 if api == "omp" else GRID, linewidth=1.4, zorder=2,
                     linestyle="-" if api == "omp" else (0, (4, 2)))
            for x, y in ser:
                col = SLOTS[(x - 1) % len(SLOTS)]
                if api == "omp":
                    eff.plot(x, y, marker="o", markersize=6, color=col, markeredgecolor=SURFACE,
                             markeredgewidth=1.0, zorder=4)
                else:
                    eff.plot(x, y, marker="D", markersize=10, markerfacecolor=SURFACE, markeredgecolor=col,
                             markeredgewidth=1.6, zorder=3)
        eff.set_xticks(range(1, 9))
        eff.set_xlim(0.5, 8.5)
        eff.set_ylim(60, 102)
        eff.axhline(100, color=ROOF, linewidth=1.2, linestyle=(0, (5, 3)))
        eff.set_xlabel("polynomial degree p")
        if eff is grid[1][0]:
            eff.set_ylabel("% of the roof (GDoF/s)")
        eff.grid(True, axis="y", color=GRID, linewidth=0.8)
        eff.set_axisbelow(True)
        for side in ("top", "right"):
            eff.spines[side].set_visible(False)
        eff.set_title(title.split(" (")[0] + " — share of the memory roof reached", loc="left", color=TEXT,
                      fontsize=10.5)
        ax.set_xlabel("arithmetic intensity (flop / byte of in + out + geometric data)")
        if ax is axes[0]:
            ax.set_ylabel("GFLOP/s")
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", color=TEXT, fontsize=11)

    # legend: colour = p, marker = API
    handles = [plt.Line2D([], [], marker="o", color=SLOTS[p - 1], markeredgecolor=SURFACE, linestyle="none",
                          markersize=8, label=f"p = {p}") for p in range(1, 9)]
    handles += [plt.Line2D([], [], marker="o", color=TEXT2, linestyle="-", markersize=6, label="OpenMP"),
                plt.Line2D([], [], marker="D", markerfacecolor=SURFACE, markeredgecolor=TEXT2, color=GRID,
                           linestyle=(0, (4, 2)), markeredgewidth=1.6, markersize=10, label="SYCL")]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, fontsize=8.5, labelcolor=TEXT2,
               bbox_to_anchor=(0.5, 0.02))
    timing = "SYCL device times" if device else "host times"
    fig.text(0.5, 0.005, f"Arc A770, icpx 2026.1, float; fastest variant per degree at the largest size of "
             f"each sweep (1e8 DoFs), {timing}; labels: p", ha="center", va="bottom", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
