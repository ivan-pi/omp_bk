# Benchmark launcher & plotting scripts

Helpers for sweeping the bake-off kernels (`BK1`, `BK3`, `BK5`) over a range of
problem sizes and plotting the results. Collecting and plotting are separate
steps.

| File                | Role                                                          |
|---------------------|---------------------------------------------------------------|
| `run_benchmarks.sh` | Sweep a BK kernel over a log-spaced DoF range into a data file. |
| `logspace.cpp`      | Portable C++ helper (`scripts/logspace`) emitting the log-spaced DoF targets. |
| `plot_bk.gp`        | gnuplot: `GDoF/s`- and `GB/s`-vs-DoF figures from that data file. |
| `plot_stream.gp`    | gnuplot: bandwidth-vs-array-size figures from `bkstream` output. |
| `bk_style.gp`       | Shared gnuplot styling, loaded by the plotting scripts.       |

`make` (from the repo root) builds the kernels and `scripts/logspace`, which
`run_benchmarks.sh` needs. `logspace` is plain ISO C++ (no OpenMP /
`-march=native`), so `make scripts/logspace` works without the kernels' offload
toolchain.

## 1. Collect — `run_benchmarks.sh`

```
scripts/run_benchmarks.sh [options] <executable> [dof_min] [dof_max] [degree]
```

Picks problem sizes so the total DoF count (`ndof = nelmt * dofs_per_element`)
is log-spaced across `[dof_min, dof_max]` (default `1e4 .. 1e8`), runs the kernel
at each, and appends a row to `results/<kernel>.dat`. With `degree` set, only
that order runs; otherwise all supported orders are scanned. Each kernel takes
the polynomial order `p` (1..8) and carries `(p+1)^3` DoFs per element:

| Kernel      | First CLI argument   | Supported | Quadrature points | DoFs / element |
|-------------|----------------------|-----------|-------------------|----------------|
| `BK1`,`BK3` | polynomial order `p` | 1..8      | `nq = p + 2`      | `(p+1)^3`      |
| `BK5`       | polynomial order `p` | 1..8      | `nq = p + 1`      | `(p+1)^3`      |

| Flag        | Meaning                                            | Default   |
|-------------|----------------------------------------------------|-----------|
| `-n N`      | number of log-spaced DoF sample points             | `12`      |
| `-t NTESTS` | timing repetitions handed to the kernel            | `5`       |
| `-o DIR`    | output directory for the `.dat` file               | `results` |
| `-h`        | help                                               |           |

```sh
scripts/run_benchmarks.sh ./BK5                        # all orders, DoFs 1e4..1e8
scripts/run_benchmarks.sh -n 20 ./BK1 1e4 1e8 3        # order 3 only, 20 points
scripts/run_benchmarks.sh -n 5 -t 2 ./BK5 1e3 1e5 4    # quick dev run
```

The full `1e4 .. 1e8` sweep allocates several GB per run at the larger orders;
use a smaller range and `-n`/`-t` while developing.

## 2. Plot — `plot_bk.gp`

A standalone gnuplot script driven by command-line arguments (`gnuplot -c`); its
only shell-out is a one-line `awk` reading the legend metadata from the header.

```
gnuplot -c scripts/plot_bk.gp <datafile> [format] [outdir]
```

| Argument   | Meaning                                            | Default                |
|------------|----------------------------------------------------|------------------------|
| `datafile` | `results/<kernel>.dat` from `run_benchmarks.sh`    | (required)             |
| `format`   | `pngcairo` or `svg`                                | `pngcairo`             |
| `outdir`   | directory for the figures                          | the datafile's folder  |

It writes two figures vs number of DoFs (log x-axis), one curve per order
(labelled `p = ...`):

- `<kernel>_gdofs.<ext>` — throughput, **GDoF/s** vs `ndof`
- `<kernel>_gbs.<ext>`   — effective bandwidth, **GB/s** vs `ndof`

```sh
gnuplot -c scripts/plot_bk.gp results/BK5.dat       # -> results/BK5_gdofs.png, _gbs.png
gnuplot -c scripts/plot_bk.gp results/BK5.dat svg   # SVG instead
```

Requires `gnuplot` (e.g. `apt-get install gnuplot-nox`).

## 3. Plot — `plot_stream.gp`

Companion to `bkstream`, which sweeps per-array sizes itself and writes one
`index` block per streaming kernel. Renders effective bandwidth vs array size
(log x-axis), one curve per kernel:

```sh
./bkstream -r 1K:256M -n 16 all > results/bkstream.dat
gnuplot -c scripts/plot_stream.gp results/bkstream.dat   # -> results/bkstream_bw.png
```

Its columns are `nelmt  ndof  bytes  time_s  gbytes_per_s`, and the header lists
the kernels (`# kernels = init copy triad striad`) for the legend.

### `bk_style.gp` and the gnuplot search path

`plot_bk.gp` keeps its axis/key styling in `bk_style.gp`, found by prepending its
own directory to the gnuplot loadpath, so the calls above work from any
directory. gnuplot also searches `$GNUPLOT_LIB` for `load`ed files, so you can
reuse the style elsewhere:

```sh
export GNUPLOT_LIB=/path/to/omp_bk/scripts
gnuplot -e "load 'bk_style.gp'; plot ..."
```

## Data file format

One file per kernel: a comment header with the metadata the plotting script
needs, then each degree as a gnuplot `index` block, blocks separated by two
blank lines:

```
# kernel = BK5
# key = p
# degrees = 1 2 3 4 5 6 7 8
# columns: ndof  nelmt  gdof_per_s  gbytes_per_s

# p = 1  (dofs_per_element=8)
1000 125 0.00522952 0.167345
...
```

Being `index`-separated, they also plot by hand, e.g.
`plot for [i=0:7] 'results/BK5.dat' index i using 1:3 with linespoints`.
