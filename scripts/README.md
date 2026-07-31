# Benchmark launcher & plotting scripts

Helpers for sweeping the bake-off kernels (`BK1`, `BK3`, `BK5`) across a range
of problem sizes and turning the results into figures.

| Script                | Role                                                        |
|-----------------------|-------------------------------------------------------------|
| `run_benchmarks.sh`   | Launcher: sweep a kernel over a log-spaced range of DoFs.   |
| `plot_results.sh`     | Post-process the `.dat` output with awk + gnuplot into PNG/SVG figures. |

Build the kernels first (`make` from the repository root).

## `run_benchmarks.sh`

```
scripts/run_benchmarks.sh [options] <executable> [dof_min] [dof_max] [degree]
```

The launcher picks a set of problem sizes so that the **total number of degrees
of freedom** (`ndof = nelmt * dofs_per_element`) is spaced logarithmically
across `[dof_min, dof_max]` (default `1e4 .. 1e8`). For every sample it runs the
kernel, parses the reported `GDoF/s` and `GB/s` with awk, and appends a row to a
per-degree data file `results/<exe>_deg<NN>.dat`.

If `degree` is given, only that degree is run; otherwise **all supported degrees
are scanned**. The per-kernel argument convention is detected from the
executable name:

| Kernel      | First CLI argument | Supported | DoFs / element |
|-------------|--------------------|-----------|----------------|
| `BK1`,`BK3` | polynomial order `p` | 1..8    | `(p+1)^3`      |
| `BK5`       | `nq` directly        | 2..8    | `nq^3`         |

Options:

| Flag        | Meaning                                            | Default   |
|-------------|----------------------------------------------------|-----------|
| `-n N`      | number of log-spaced DoF sample points             | `12`      |
| `-t NTESTS` | timing repetitions handed to the kernel            | `5`       |
| `-o DIR`    | output directory for the `.dat` files              | `results` |
| `-p`        | run `plot_results.sh` on the output when finished  | off       |
| `-h`        | help                                               |           |

### Examples

```sh
# Full production sweep of BK5, all nq, DoFs 1e4..1e8, then plot:
scripts/run_benchmarks.sh -p ./BK5

# BK1 at polynomial order 3 only:
scripts/run_benchmarks.sh ./BK1 1e4 1e8 3

# Quick development run: small range, few points, few repetitions:
scripts/run_benchmarks.sh -n 5 -t 2 ./BK5 1e3 1e5 4
```

> **Development note.** The full `1e4 .. 1e8` sweep allocates several GB per run
> at the larger degrees and takes a while. While developing, pass a smaller DoF
> range and a small `-n`/`-t` (as in the last example above) to keep runs fast
> and memory-light.

## `plot_results.sh`

```
scripts/plot_results.sh [-o DIR] [-f FORMAT] <executable-or-base>
```

Reads `DIR/<base>_deg*.dat` and renders two figures against the number of DoFs
(logarithmic x-axis), one curve per degree:

- `<base>_gdofs.<ext>` — throughput, **GDoF/s** vs `ndof`
- `<base>_gbs.<ext>`   — effective bandwidth, **GB/s** vs `ndof`

| Flag       | Meaning                                | Default    |
|------------|----------------------------------------|------------|
| `-o DIR`   | directory with the `.dat` files / output | `results` |
| `-f FORMAT`| `pngcairo` or `svg`                    | `pngcairo` |

Requires `gnuplot` (e.g. `apt-get install gnuplot-nox`).

## Data file format

Each `.dat` file carries a two-line header followed by whitespace-separated
columns:

```
# kernel=BK5  mode=nq  degree=4  dofs_per_element=64
# ndof  nelmt  gdof_per_s  gbytes_per_s
1024 16 0.0024072 0.0770303
...
```
