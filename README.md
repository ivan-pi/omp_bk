# omp_bk

OpenMP `target`-offload implementations of the CEED "bake-off" kernels
(sum-factorized spectral-element operators):

| Program | Operator                                   |
|---------|--------------------------------------------|
| `BK1`   | Mass matrix                                |
| `BK3`   | Poisson (stiffness) matrix                 |
| `BK5`   | Collocated Laplacian at quadrature points  |
| `BK1_ff`| Mass matrix in float-float arithmetic (see below) |

## Requirements

- A C++17 compiler with OpenMP support (e.g. GCC or Clang).
- To run the kernels on a GPU, an OpenMP-offload-capable toolchain
  (e.g. GCC built with `nvptx`/`amdgcn` offload, or Clang with the matching
  offload runtime). Without a GPU the `target` regions run as a host fallback.

## Build

```sh
make            # builds BK1, BK3, BK5, BK1_ff
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

## Float-float version of BK1

`BK1_ff` runs the BK1 kernel of `bk1_sumfact.h` with the arithmetic type
`ffloat` from `float_float.h`: a value is the unevaluated sum of two floats
(~48 significant bits, float's exponent range), and every `+` and `*` is a
short chain of fp32 adds, multiplies and fused multiply-adds built on the
error-free transformations TwoSum and TwoProd. On GPUs whose fp64 rate is
1/32 or 1/64 of fp32, this recovers near-double accuracy at about 18 fp32
flops per multiply-add while moving the same 8 bytes per value as double.
The kernel source is unchanged: it is the same template that `BK1`
instantiates with `float`.

The driver runs double, float and float-float on the same data, prints one
row per precision and the relative L2 error of float (~1e-7) and float-float
(~3e-15) against the double result, all reduced in double. The last line is
the check of the float-float result, `ok` or `FAIL` against a tolerance
(default 1e-10, `BK_TOL=...`), with exit status 1 on failure:

```sh
./BK1_ff <p> [nelmt] [ntests]     # same arguments and constant data as BK1; the float row prints BK1's norm
BK_RANDOM=1 ./BK1_ff 2            # pseudo-random data (the same as BK1_amx's)
for p in 1 2 3 4 5 6 7 8; do ./BK1_ff $p 4096 1 | tail -1; done   # the check per order
```

The error-free transformations need strict IEEE semantics: the header
refuses `-ffast-math`, anything subtler shows up as `FAIL`, and the Makefile
selects the native instruction set so `std::fma` is an instruction (the
first output line says `fma = hardware`).

## Apple AMX version of BK1

`amx/BK1_amx.cpp` runs the BK1 sum factorization on the Apple AMX
coprocessor; elsewhere `amx/amx.h` emulates the AMX instructions in
software, so the kernel logic can be checked on any machine. The same binary
also runs the CPU baselines (`BK_KERNEL=neon`: 4-wide NEON FMAs across the
elements of a batch; `BK_KERNEL=refv`: the reference with unit-stride inner
loops). The header of `amx/BK1_amx.cpp` lists all environment variables.

```sh
make BK1 BK1_amx bw_test amx_pipe     # accel_gemm needs macOS (Accelerate)
./BK1_amx <p> [nelmt] [ntests]        # same arguments as BK1, p = 1..14
BK_KERNEL=neon BK_LAYOUT=soa ./BK1_amx 4

amx/scripts/validate.sh               # every kernel against the reference; exit 1 on failure
amx/scripts/throughput.sh results.csv # sweep (all kernels from BK1_amx): kernel,p,target,nelmt,dofs,gdofs,gbs
amx/scripts/high_order.sh results_high.csv   # the same for p = 9..14 (AMX binary only)
OUT=tune.csv amx/scripts/tune.sh      # best layout/path/batch per order
./bw_test    > bw.csv                 # bandwidth and FMA roofs: test,threads,bytes,seconds,GB_per_s
./amx_pipe   > pipe.txt               # AMX load-to-use and store behaviour (one unit)
./accel_gemm > gemm.csv               # Accelerate sgemm calibration (macOS)

amx/scripts/plot_throughput.py results.csv throughput.png   # per kernel and order, AMX/NEON speedup
amx/scripts/plot_roofline.py results.csv bw.csv roofline.png # plateaus against the roofs
```

The scripts find `BK1` and `BK1_amx` in the repository root and run from
any directory; set `BK1_AMX=...` or `BK1=...` to use other builds. The
plots need matplotlib.
