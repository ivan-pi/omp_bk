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

## OpenACC versions

`acc/BK1_acc.cpp`, `acc/BK3_acc.cpp` and `acc/BK5_acc.cpp` are the same
kernels with OpenACC directives: a plain `acc parallel loop` over the
elements, with the inner loop nests left to the compiler. They take the same
arguments as the OpenMP executables and print the same two lines, so the
scripts' parsing carries over. `BK_RANDOM=1` replaces the constant test data
with seeded random data.

```sh
make -C acc                        # nvc++ -acc=gpu, prints the -Minfo=acc report
make -C acc ACC="-acc=gpu -gpu=cc80"
make -C acc ACC=-acc=multicore     # OpenACC on the host cores
make -C acc host                   # BK?_acc_host: serial host build (-acc=host) for validate.sh
make -C acc CXX=g++                # GCC -fopenacc (host fallback without an offload toolchain)
./acc/BK1_acc <p> [nelmt] [ntests]

make BK1 BK3 BK5 && make -C acc all host
acc/scripts/validate.sh            # every order: constant data against BK1/BK3/BK5,
                                   # random data against the host build; exit 1 on failure
```

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
