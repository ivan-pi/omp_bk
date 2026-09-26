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

## Apple AMX version of BK1

`amx/BK1_amx.cpp` runs the BK1 sum factorization on the Apple AMX
coprocessor (Apple Silicon); elsewhere the AMX instructions are emulated in
software by `amx/amx.h`, so the kernel logic can be checked on any machine.

```sh
make BK1_amx                         # or: make CXX=clang++ BK1_amx
./BK1_amx <p> [nelmt] [ntests]       # same arguments as BK1, p = 1..14
```

The same binary also runs the CPU baselines on the same data:
`BK_KERNEL=neon` uses 4-wide NEON FMAs (GNU vector types) across the
elements of a batch, and `BK_KERNEL=refv` is the reference with its loops
interchanged for unit-stride inner loops. The header of `amx/BK1_amx.cpp`
lists all environment variables (kernel, layout, dense path, batch size,
random test data) and `amx/scripts/` holds the validation, tuning and
throughput-sweep scripts, which run the `BK1` and `BK1_amx` executables
built by `make` in the repository root:

```sh
make BK1 BK1_amx
amx/scripts/validate.sh          # every kernel against the reference, random data
amx/scripts/throughput.sh        # CEED-style throughput sweep to results.csv
```

`amx/bench/` holds the micro-benchmarks that give the roofs for those
results: `bw_test` (STREAM-style bandwidth, NEON and AMX FMA peaks),
`amx_pipe` (the AMX unit's load/store pipeline) and `accel_gemm`
(Accelerate's sgemm as a calibration point, macOS only).

```sh
make bw_test amx_pipe
./bw_test > bw.csv
amx/scripts/plot_roofline.py results.csv bw.csv     # plateaus against the roofs
```
