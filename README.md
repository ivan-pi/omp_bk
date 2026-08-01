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

## Arithmetic intensity

The kernel bodies live in headers (`bk1_kernel.h`, `bk3_kernel.h`,
`bk5_kernel.h`) so their flop count can be measured directly rather than
estimated. `arith_intensity` instantiates each kernel with an
operation-counting scalar type (`flop_counter.h`) and runs it for a single
element: every `+`/`*` the kernel executes bumps a counter, giving the exact
flops. Bytes are the streaming DRAM traffic per element (read `in`, write
`out`, read the geometric factors), matching the effective `GB/s` the timing
drivers report.

```sh
make arith_intensity
./arith_intensity          # flops, bytes, flop/byte and byte/flop per order
```

The measured counts are asserted against closed-form formulas, so the tool
also self-checks; it exits non-zero on any mismatch. `arith_intensity.py`
reproduces the same formulas without compiling and includes an experiment
showing that the contraction order is irrelevant when the order `p` is equal
in all three directions, but not otherwise.
