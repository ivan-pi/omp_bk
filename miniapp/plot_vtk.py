#!/usr/bin/env python3
"""plot_vtk.py -- plot the legacy VTK files written by `bp --vtk`.

usage: ./plot_vtk.py solution.vtk [out.png]
       ./plot_vtk.py heat_0000.vtk heat_0001.vtk ... [out.png]

One file: the mid-plane (z = 1/2) slice of u and of the error u - u_exact
on the node coordinates (so a warped mesh shows its real geometry), and the
profile of u along the centre line y = z = 1/2 against the exact solution.
Several files (a heat run): the centre-line profile of every file, light to
dark in time, with the exact solution of the last one, and max |error|
against the step.

Only numpy and matplotlib are needed; the reader handles exactly what
bp_vtk.h writes (ASCII STRUCTURED_GRID with POINT_DATA scalars).
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_vtk(path):
    """Return (coords[nx, ny, nz, 3], {name: field[nx, ny, nz]})."""
    with open(path) as f:
        tokens = f.read().split()
    i = tokens.index("DIMENSIONS")
    nx, ny, nz = (int(t) for t in tokens[i + 1:i + 4])
    n = nx * ny * nz
    i = tokens.index("POINTS")
    pts = np.array(tokens[i + 3:i + 3 + 3 * n], dtype=float).reshape(n, 3)
    # VTK order: x fastest -> array index (z, y, x); transpose to (x, y, z)
    coords = pts.reshape(nz, ny, nx, 3).transpose(2, 1, 0, 3)
    fields = {}
    j = i + 3 + 3 * n
    while "SCALARS" in tokens[j:]:
        j = tokens.index("SCALARS", j)
        name = tokens[j + 1]
        start = j + 6   # SCALARS name type 1 LOOKUP_TABLE default
        vals = np.array(tokens[start:start + n], dtype=float)
        fields[name] = vals.reshape(nz, ny, nx).transpose(2, 1, 0)
        j = start + n
    return coords, fields


def slice_mid(coords, field):
    """x, y and the values on the z = 1/2 node plane."""
    k = coords.shape[2] // 2
    return coords[:, :, k, 0], coords[:, :, k, 1], field[:, :, k]


def centre_line(coords, field):
    """x and the values along the line y = z = 1/2."""
    j = coords.shape[1] // 2
    k = coords.shape[2] // 2
    return coords[:, j, k, 0], field[:, j, k]


def plot_one(path, out):
    coords, fields = read_vtk(path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    x, y, u = slice_mid(coords, fields["u"])
    m = axes[0].pcolormesh(x, y, u, cmap="Blues", shading="gouraud")
    axes[0].set_title("u on z = 1/2")
    fig.colorbar(m, ax=axes[0])

    x, y, e = slice_mid(coords, fields["error"])
    emax = max(np.abs(e).max(), 1e-300)
    m = axes[1].pcolormesh(x, y, e, cmap="coolwarm", vmin=-emax, vmax=emax, shading="gouraud")
    axes[1].set_title("u - u_exact on z = 1/2")
    fig.colorbar(m, ax=axes[1], format="%.1e")
    for ax in axes[:2]:
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # the node grid, to show the elements and any warp
        ax.plot(x, y, color="white", linewidth=0.3, alpha=0.6)
        ax.plot(x.T, y.T, color="white", linewidth=0.3, alpha=0.6)

    xl, ul = centre_line(coords, fields["u"])
    _, el = centre_line(coords, fields["u_exact"])
    axes[2].plot(xl, el, color="#555555", linewidth=1.5, label="exact")
    axes[2].plot(xl, ul, "o", color="#2a78d6", markersize=4, label="bp nodes")
    axes[2].set_title("profile along y = z = 1/2")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("u")
    axes[2].legend(frameon=False)

    fig.suptitle(path)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_series(paths, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ramp = plt.get_cmap("Blues")
    errors = []
    for n, path in enumerate(paths):
        coords, fields = read_vtk(path)
        xl, ul = centre_line(coords, fields["u"])
        shade = 0.35 + 0.65 * n / max(1, len(paths) - 1)
        axes[0].plot(xl, ul, color=ramp(shade), linewidth=1.5,
                     label=f"step {n}" if n in (0, len(paths) - 1) else None)
        errors.append(np.abs(fields["error"]).max())
    _, el = centre_line(coords, fields["u_exact"])
    axes[0].plot(xl, el, "--", color="#555555", linewidth=1.2, label="exact, last step")
    axes[0].set_title("u along y = z = 1/2, light to dark in time")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend(frameon=False)

    axes[1].plot(range(len(paths)), errors, "o-", color="#2a78d6", markersize=4)
    axes[1].set_title("max |u - u_exact|")
    axes[1].set_xlabel("step")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    args = argv[1:]
    out = None
    if args[-1].endswith(".png"):
        out = args.pop()
    if not args:
        print(__doc__)
        return 1
    if len(args) == 1:
        plot_one(args[0], out or args[0].rsplit(".", 1)[0] + ".png")
    else:
        plot_series(args, out or args[0].rsplit("_", 1)[0] + "_series.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
