# Benchmark launcher & plotting scripts

Helpers for sweeping the bake-off kernels (`BK1`, `BK3`, `BK5`) across a range
of problem sizes and turning the results into figures. Collecting and plotting
are deliberately kept as two independent steps.

| File                | Role                                                          |
|---------------------|---------------------------------------------------------------|
| `run_benchmarks.sh` | Launcher: sweep a kernel over a log-spaced range of DoFs, writing a column data file. |
| `plot_bk.gp`        | gnuplot script: render `GDoF/s`- and `GB/s`-vs-DoF figures from that data file. |
| `bk_style.gp`       | Shared gnuplot axis/key styling, loaded by `plot_bk.gp`.       |

Build the kernels first (`make` from the repository root).

## 1. Collect — `run_benchmarks.sh`

```
scripts/run_benchmarks.sh [options] <executable> [dof_min] [dof_max] [degree]
```

The launcher picks a set of problem sizes so that the **total number of degrees
of freedom** (`ndof = nelmt * dofs_per_element`) is spaced logarithmically
across `[dof_min, dof_max]` (default `1e4 .. 1e8`). For every sample it runs the
kernel, parses the reported `GDoF/s` and `GB/s`, and appends a row to a single
per-kernel data file `results/<kernel>.dat`.

If `degree` is given, only that order is run; otherwise **all supported orders
are scanned**. Every kernel takes the polynomial order `p` (1..8) as its first
argument and carries `(p+1)^3` DoFs per element:

| Kernel      | First CLI argument   | Supported | Quadrature points | DoFs / element |
|-------------|----------------------|-----------|-------------------|----------------|
| `BK1`,`BK3` | polynomial order `p` | 1..8      | `nq = p + 2`      | `(p+1)^3`      |
| `BK5`       | polynomial order `p` | 1..8      | `nq = p + 1`      | `(p+1)^3`      |

Options:

| Flag        | Meaning                                            | Default   |
|-------------|----------------------------------------------------|-----------|
| `-n N`      | number of log-spaced DoF sample points             | `12`      |
| `-t NTESTS` | timing repetitions handed to the kernel            | `5`       |
| `-o DIR`    | output directory for the `.dat` file               | `results` |
| `-h`        | help                                               |           |

### Examples

```sh
# Full production sweep of BK5, all nq, DoFs 1e4..1e8:
scripts/run_benchmarks.sh ./BK5

# BK1 at polynomial order 3 only, 20 sample points:
scripts/run_benchmarks.sh -n 20 ./BK1 1e4 1e8 3

# Quick development run: small range, few points, few repetitions:
scripts/run_benchmarks.sh -n 5 -t 2 ./BK5 1e3 1e5 4
```

> **Development note.** The full `1e4 .. 1e8` sweep allocates several GB per run
> at the larger degrees and takes a while. While developing, pass a smaller DoF
> range and a small `-n`/`-t` (as in the last example above) to keep runs fast
> and memory-light.

## 2. Plot — `plot_bk.gp`

Plotting is a standalone gnuplot script driven with command-line arguments
(`gnuplot -c`); it does not shell out except for a one-line inline `awk` call to
read the legend metadata out of the data-file header.

```
gnuplot -c scripts/plot_bk.gp <datafile> [format] [outdir]
```

| Argument   | Meaning                                            | Default                |
|------------|----------------------------------------------------|------------------------|
| `datafile` | `results/<kernel>.dat` from `run_benchmarks.sh`    | (required)             |
| `format`   | `pngcairo` or `svg`                                | `pngcairo`             |
| `outdir`   | directory for the figures                          | the datafile's folder  |

It writes two figures against the number of DoFs (logarithmic x-axis), one curve
per polynomial order (labelled `p = ...`):

- `<kernel>_gdofs.<ext>` — throughput, **GDoF/s** vs `ndof`
- `<kernel>_gbs.<ext>`   — effective bandwidth, **GB/s** vs `ndof`

```sh
scripts/run_benchmarks.sh ./BK5                     # -> results/BK5.dat
gnuplot -c scripts/plot_bk.gp results/BK5.dat       # -> results/BK5_gdofs.png, results/BK5_gbs.png
gnuplot -c scripts/plot_bk.gp results/BK5.dat svg   # SVG instead
```

Requires `gnuplot` (e.g. `apt-get install gnuplot-nox`).

### `bk_style.gp` and the gnuplot search path

`plot_bk.gp` factors its axis/key styling into `bk_style.gp` and finds it by
prepending its own directory to the gnuplot loadpath, so the invocations above
work from any current directory. gnuplot also searches `$GNUPLOT_LIB` for
`load`ed files, so you can reuse the style from your own scripts by putting this
folder on that path:

```sh
export GNUPLOT_LIB=/path/to/omp_bk/scripts
gnuplot -e "load 'bk_style.gp'; plot ..."
```

## Data file format

One file per kernel. A short comment header carries the metadata the plotting
script needs; each degree is a separate gnuplot `index` block, blocks separated
by two blank lines:

```
# kernel = BK5
# key = p
# degrees = 1 2 3 4 5 6 7 8
# columns: ndof  nelmt  gdof_per_s  gbytes_per_s

# p = 1  (dofs_per_element=8)
1000 125 0.00522952 0.167345
...


# p = 2  (dofs_per_element=27)
999 37 0.00504739 0.161517
...
```

Because the datasets are `index`-separated, they can also be plotted by hand,
e.g. `plot for [i=0:7] 'results/BK5.dat' index i using 1:3 with linespoints`.
