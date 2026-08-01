# omp_bk

OpenMP `target`-offload implementations of the CEED "bake-off" kernels
(sum-factorized spectral-element operators):

| Program | Operator                                   |
|---------|--------------------------------------------|
| `BK1`   | Mass matrix                                |
| `BK3`   | Poisson (stiffness) matrix                 |
| `BK5`   | Collocated Laplacian at quadrature points  |

A companion program, `bkstream`, provides STREAM-style bandwidth
micro-benchmarks (init/copy/triad/Schönauer-triad) over the same DoF data
layout and `target teams loop` harness, as a peak-bandwidth reference.

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
./BK1 <p> [nelmt] [ntests]      # p is the polynomial order (1..8)
./BK3 <p> [nelmt] [ntests]      #   BK1/BK3: nq = p + 2 quadrature points
./BK5 <p> [nelmt] [ntests]      #   BK5 is collocated: nq = p + 1
```

All three take the polynomial order `p` and carry `(p+1)^3` DoFs per element.

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

## Streaming bandwidth reference (`bkstream`)

`bkstream` measures achievable memory bandwidth with STREAM-style kernels laid
out and timed exactly like the BK operators (a flat `std::vector` of shape
`nelmt*nm*nm*nm` with `nm = p + 1`, the same `target teams loop`, and the same
min-over-repetitions timing). It sweeps a logarithmic range of per-array sizes
itself and writes a gnuplot data file — one `index` block per kernel:

```sh
./bkstream copy                                   # one kernel, default 1K..64M range
./bkstream -r 1K:256M -n 16 all > results/bkstream.dat   # init, copy, triad, striad
gnuplot -c scripts/plot_stream.gp results/bkstream.dat   # -> results/bkstream_bw.png
```

Kernels: `init` (`Y = a`), `copy` (`Y = X`), `triad` (`Y = X + a*Z`), and
`striad` (Schönauer, `Y = X + Z*W`). Sizes accept `B`/`K`/`M`/`G` suffixes
(base 1024). Run `./bkstream -h` for all options. Unlike the BK operators these
kernels do not stream a geometric-factor array, so the numbers form an upper
reference for the BK effective bandwidth. Columns of the data file are
`nelmt  ndof  bytes  time_s  gbytes_per_s`.
