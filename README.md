# omp_bk

OpenMP `target`-offload implementations of the CEED "bake-off" kernels
(sum-factorized spectral-element operators):

| Program | Operator                                   |
|---------|--------------------------------------------|
| `BK1`   | Mass matrix                                |
| `BK3`   | Poisson (stiffness) matrix                 |
| `BK5`   | Collocated Laplacian at quadrature points  |

## Requirements

- A C++17 compiler with OpenMP support (e.g. GCC or Clang).
- To run the kernels on a GPU, an OpenMP-offload-capable toolchain
  (e.g. GCC built with `nvptx`/`amdgcn` offload, or Clang with the matching
  offload runtime). Without a GPU the `target` regions run as a host fallback.

## Build

```sh
make            # builds BK1, BK3, BK5
make clean      # removes the executables
```

The default flags live in `Makefile` (`CXX`, `CXXFLAGS`). Override them from
the command line, e.g. to use Clang:

```sh
make CXX=clang++
```

## Run

```
./BK1 <p> [nelmt] [ntests]      # BK1/BK3: p is the polynomial order, nq = p + 2
./BK3 <p> [nelmt] [ntests]
./BK5 <nq> [nelmt] [ntests]     # BK5: first argument is nq directly
```

- `nelmt`  — number of elements (default 524288).
- `ntests` — timing repetitions; the minimum wall time is reported (default 5).

Each run prints the achieved `GDoF/s` and effective `GB/s`, followed by the
solution norm (useful as a quick correctness check).

## Benchmark sweeps

Collecting and plotting are two separate steps.
`scripts/run_benchmarks.sh` sweeps a kernel over a logarithmic range of DoF
counts (default `1e4 .. 1e8`) into a per-kernel data file, and the gnuplot
script `scripts/plot_bk.gp` turns that file into `GDoF/s`- and `GB/s`-vs-DoF
figures:

```sh
scripts/run_benchmarks.sh ./BK5                 # sweep all degrees -> results/BK5.dat
gnuplot -c scripts/plot_bk.gp results/BK5.dat   # -> results/BK5_gdofs.png, results/BK5_gbs.png

scripts/run_benchmarks.sh -n 20 ./BK1 1e4 1e8 3 # single order, 20 sample points
```

See [`scripts/README.md`](scripts/README.md) for the full options and the
per-kernel argument conventions.
