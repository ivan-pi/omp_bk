# omp_bk

Implementations of the CEED "bake-off" kernels (sum-factorized
spectral-element operators) in OpenMP `target` offload, SYCL and the Intel
Tiny Tensor Compiler (tinytc), plus an Apple AMX version of BK1:

| Program | Operator                                   |
|---------|--------------------------------------------|
| `BK1`   | Mass matrix                                |
| `BK3`   | Poisson (stiffness) matrix                 |
| `BK5`   | Collocated Laplacian at quadrature points  |
| `BK1_ff`| Mass matrix in float-float arithmetic (see below) |

## Layout

The tree is organized by approach; the file names say the kernel.

| Directory | Contents |
|-----------|----------|
| `include/` | Shared across approaches: `bk_common.h` (test data, command line, order dispatch, timing), `bk_harness.h` (references, layouts and checks of the GPU harnesses), `bk_sycl.h` (SYCL timing), `bk1_gpu.h` and `bk1_gpu_steps.h` (BK1 GPU helpers used by OpenMP offload and SYCL) |
| `openmp/` | The portable drivers `BK1.cpp`, `BK3.cpp`, `BK5.cpp`, `BK1_ff.cpp` (host or offload), `bk1_sumfact.h`, `bk3_sumfact.h`, `float_float.h` |
| `openmp/offload/` | GPU kernel variants for BK1 (`bk1_offload.cpp`, one header per strategy in `bk1_offload/`), BK3 (`bk3_offload.cpp`) and BK5 (`bk5_offload.cpp`) |
| `sycl/` | `bk1_sycl.cpp`, `bk3_sycl.cpp`, `bk5_sycl.cpp` |
| `tinytc/` | `bk_tinytc.cpp` (driver for BK1, BK3 and BK5), `ir/` (the kernels; `bk1.ir` and `bk5.ir` as received, `bk3_blk.ir` and `bk3_ec.ir` in tinytc's blocked and our layout, `bk3_*_gs.ir` with S in global memory), `repro_slm/` (bug reproducer), `upstream_issues.md` (drafts) |
| `cm/` | `bk1_cm_genx.cpp`, `bk5_cm_genx.cpp` (BK1 and BK5 in C for Metal), `bk1_cm.cpp`, `bk5_cm.cpp` (SYCL hosts), `cm_common.h`, `cm_sycl.hpp`, `Makefile`, `README.md` |
| `amx/` | `BK1_amx.cpp`, `amx.h` (AMX instructions, emulated off Apple Silicon), `scripts/` |
| `roofline/` | Micro-benchmarks for the roofs: `stream_omp.cpp`, `stream_sycl.cpp` (GPU), `bw_test.cpp`, `amx_pipe.cpp`, `accel_gemm.cpp` (CPU, AMX, macOS) |
| `scripts/` | Intel GPU build, run and environment; `study/` sweeps and surveys, `plot/`, `profile/` (Advisor, VTune) |
| `results/<machine>/` | Measured CSVs, tables and plots (`a770/`: Intel Arc A770) |

## Requirements

- CMake 3.21 or later and a C++17 compiler, with OpenMP for threads (e.g. GCC
  or Clang), for the host build; there the `target` regions run on the host.
- For the GPU builds: the Intel oneAPI DPC++/C++ compiler (icpx, which
  provides both OpenMP offload and SYCL), Ninja, and an Intel GPU with the
  Level Zero driver. Optional: the Tiny Tensor Compiler (tinytc) for
  `bk_tinytc`.

## Build

One build tree per backend: the presets `host` (CPU, any compiler),
`gpu-openmp` and `gpu-sycl` (Intel GPU, icpx). Their options:

| Option | Default | Builds |
|---|---|---|
| `BK_ENABLE_OPENMP` | `ON` (`gpu-openmp`) | host: OpenMP threads (off: serial); GPU: the OpenMP offload programs (`BK1_<p>`, `bk1_offload_<p>`, ...) |
| `BK_ENABLE_SYCL` | `ON` (`gpu-sycl`) | GPU: the SYCL programs (`bk1_sycl_<p>`, ...) |
| `BK_ENABLE_TINYTC` | `OFF` | GPU: `bk_tinytc`; needs SYCL and a tinytc install (`TINYTC_ROOT`) |
| `BK_GPU_DEVICE` | `acm-g10` in the GPU presets | ocloc device of the GPU tree (`acm-g10` = Arc A770/A750), or `jit`; empty: host tree |
| `BK_ORDERS_P`, `BK_ORDERS_BK5`, `BK_ORDERS_GPU` | `1;...;8` (BK5: `1;...;7`) | GPU: the orders p built, one binary each |
| `BK_GPU_IN_ALL` | `ON` | GPU: per-order programs in `all` (off: build them by name) |
| `BK_GPU_BASELINES` | `OFF` (`a770`: `ON`) | GPU: the one-element-per-thread variants (`bk1_sycl` `elem*` from p = 3, `bk3_offload` `ref`, `bk5_offload` `ref6i`, `ref9i`) |
| `BK_GPU_JOBS` | `0` (no limit) | GPU: compiles and links at once (Ninja), on top of `-j` |

**Which `BK_GPU_DEVICE`?** The GPU programs are compiled ahead of time
into machine code for one GPU, so this names your GPU as `ocloc` (Intel's
device compiler, which icpx calls) knows it. There is no generic value;
the preset's `acm-g10` is the Arc A770/A750 (Alchemist ACM-G10 die, Xe-HPG).
To find yours, run `sycl-ls`, which prints each GPU with its IP version:

```
[level_zero:gpu][level_zero:0] ... Intel(R) Arc(TM) A770 Graphics 12.55.8 [...]
```

That version (`12.55.8`) can be passed as it is (`-DBK_GPU_DEVICE=12.55.8`),
and `ocloc ids <name>` shows which version a name stands for (`ocloc ids
acm-g10` gives 12.55.8). `ocloc compile --help` lists every name under
`-device`. Some common ones: `acm-g10` (Arc A770/A750), `acm-g11` (Arc
A380 and the A3 laptop chips), `bmg-g21` (Arc B580/B570), `pvc` (Data Center GPU Max),
`mtl-u`, `mtl-h` (Meteor Lake integrated graphics; `xe-lpg` covers both),
`lnl-m` (Lunar Lake integrated graphics). A comma-separated list
(`acm-g10,bmg-g21`) builds one binary for several GPUs, at the cost of a
longer compile per GPU. `BK_GPU_DEVICE=jit` builds SPIR-V instead, which
any Intel GPU compiles when a kernel is first launched: the build skips
ocloc (a few seconds per binary), the first run pays the device compile (1
to 3 s for a driver), later runs use the driver's kernel cache, and the
kernels run as fast as ahead-of-time ones.

**CPU (any compiler)**

```sh
cmake --preset host               # build/host with the default compiler
cmake --build build/host          # BK1, BK3, BK5, BK1_ff, BK1_amx, bw_test, amx_pipe, stream_omp
```

**Intel GPU, OpenMP offload: the BK1, BK3, BK5 drivers**

```sh
source /opt/intel/oneapi/setvars.sh
cmake --preset gpu-openmp -DBK_GPU_DEVICE=acm-g10   # build/gpu-openmp
cmake --build build/gpu-openmp --target drivers     # BK1_<p>, BK3_<p>, BK5_<p>, BK1_ff_<p>: 15 s
cmake --build build/gpu-openmp                      # plus bk{1,3,5}_offload_<p>, stream_omp: 114 s in all
```

**Intel GPU, SYCL**

```sh
source /opt/intel/oneapi/setvars.sh
cmake --preset gpu-sycl -DBK_GPU_DEVICE=acm-g10     # build/gpu-sycl
cmake --build build/gpu-sycl                        # bk{1,3,5}_sycl_<p>, bk{1,5}_sycl_<p>_jit, stream_sycl: 50 s
```

**A few orders, or tinytc**

```sh
cmake --preset gpu-sycl -B build/sycl246 -DBK_ORDERS_GPU="2;4;6"
export TINYTC_ROOT=<tinytc install>       # or -DTINYTC_ROOT=..., or the install on CMAKE_PREFIX_PATH
cmake --preset gpu-sycl -DBK_ENABLE_TINYTC=ON && cmake --build build/gpu-sycl --target bk_tinytc
```

Every GPU tree also has the targets `all_<prog>` (every order of a program,
e.g. `all_bk5_sycl`), `all_openmp`, `all_sycl` and `drivers`. Times are
from the A770 machine (16 threads, 32 GB, icpx 2026.1).

> **GPU build time.** Each GPU binary holds one polynomial order, compiled
> ahead of time for `BK_GPU_DEVICE`; there are 31 drivers, 31 OpenMP
> offload variant programs and 40 SYCL ones.
> - **No host fallback.** Almost all of the OpenMP compile time was icpx's
>   loop optimizer (HIR) working on the host fallback copy of each target
>   region (`BK3_4`: 81 s of 82 s), code that never runs on the GPU
>   (`run.sh` sets `OMP_TARGET_OFFLOAD=MANDATORY`). The GPU tree compiles
>   the OpenMP programs with `-fopenmp-offload-mandatory`, which leaves the
>   fallback out: a driver builds in 1 to 3 s, with the same GPU timings,
>   norms and errors. Such a binary stops with an error where there is no
>   GPU instead of running on the host; use the host tree for the CPU. SYCL
>   kernels have no host fallback, so the SYCL programs never had this cost;
>   their time is the device code generation at link time.
> - **Parallel jobs.** A GPU compile or link peaks at about 0.5 GB, so by
>   default Ninja runs as many as `-j` allows (all cores). `-DBK_GPU_JOBS=n`
>   caps them at n, e.g. on a machine with little memory per core (allow
>   about 1 GB per job), or when the build shares the machine with GPU runs.
>   It works with the Ninja generator only; with Makefiles use `make -j n`.
> - **Naive baselines.** One element per thread is the fastest BK1 kernel
>   at p = 1 and close behind at p = 2, but from p = 3 on it is 4 to 100
>   times slower than the batched kernels; for BK3 and BK5 it never wins
>   (about 60% of the best at p = 1, 3% and less from p = 3). Those cases
>   are off unless `BK_GPU_BASELINES=ON` (`-DBK_NO_BASELINES`): `bk1_sycl`'s
>   five `elem*` variants from p = 3 (they stay at p = 1, 2), and the `ref`
>   kernels of `bk3_offload` and `bk5_offload`. The `elem*` variants spill
>   registers heavily at high order and took most of `bk1_sycl`'s device
>   link (`bk1_sycl_6`: 65 s with them, 9 s without), so the SYCL tree
>   builds in 50 s instead of 142 s; the `ref` kernels cost 1 to 2 s each.
>   Every variant is still checked against the host reference.
> - **Build less:** fewer orders (`BK_ORDERS_*`), one backend per tree, or
>   single targets (`--target bk1_sycl_4`, `all_bk5_sycl`, `drivers`).
> - **Parallel device link** (not the default): split the device code per
>   kernel and let icpx compile the pieces in parallel at link time,
>   ```sh
>   -DBK_EXTRA_FLAGS="-fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=8 -fopenmp-device-code-split=per_kernel -fopenmp-max-parallel-link-jobs=8"
>   ```
>   This cut the build of `bk1_sycl_6` (with the baselines) from 66 s to 29 s at 2.2 GB instead
>   of 0.7 GB, with the same results; it pays off for a single program with
>   many kernels, less so when Ninja already runs many jobs.

The flags are `-Wall -pedantic -O3 -std=c++17` with the native instruction
set and OpenMP; no build type is needed (a `Release` build would add
`-DNDEBUG`, and the benchmark keeps its asserts). Another compiler or extra
flags go in at configure time, e.g.:

```sh
cmake --preset host -B build/clang -DCMAKE_CXX_COMPILER=clang++
cmake --preset host -B build/ff8 -DCMAKE_CXX_FLAGS=-DFF_PLANE_BATCH=8
```

With icpx the host tree takes minutes to build (BK1, BK3, BK5, BK1_ff: 203 s
on 16 threads, g++: 13 s), nearly all of it in icpx's loop optimizer (HIR).
`-DCMAKE_CXX_FLAGS="-mllvm -loopopt=0"` turns it off (8 s), but it changes
the speed of the host kernels both ways: on the A770 machine's CPU, BK1
lost about a third, while BK3 got 1.5 to 2.8 times and BK5 up to 2.9 times
faster (p = 2, 4, 6; one run each).

## Run

```
build/host/BK1 <p> [nelmt] [ntests]      # p: polynomial order; BK1, BK3: nq = p + 2
build/host/BK3 <p> [nelmt] [ntests]
build/host/BK5 <p> [nelmt] [ntests]      # BK5: nq = p + 1
```

- `nelmt`  — number of elements (default 524288).
- `ntests` — timing repetitions; the minimum wall time is reported (default 5).

Each run prints the achieved `GDoF/s` and effective `GB/s`, followed by the
solution norm (useful as a quick correctness check).

## Float-float version of BK1

`BK1_ff` runs the BK1 sum factorization in float-float arithmetic. The
kernel `SumFactorizationFF` in `openmp/BK1_ff.cpp` has the interface of
`SumFactorization` in double: the caller passes IEEE doubles for the basis,
the weights and the input and gets doubles back. Inside the kernel every
value is split into a pair of floats (hi + lo, ~48 significant bits, float's
exponent range) as it is loaded, the nine steps run on separate hi and lo
work arrays, and the result is joined back into a double as it is stored.
Every multiply and multiply-add is a short chain of fp32 adds, multiplies
and fused multiply-adds built on the error-free transformations TwoSum and
TwoProd; the helpers live in `float_float.h`, a private implementation
header of the driver with no pair type in any interface. On GPUs whose fp64
rate is 1/32 or 1/64 of fp32, this recovers near-double accuracy at 15 fp32
flops per multiply-add while moving the same 8 bytes per value as double
(the split and join cost one fp64 operation per value loaded or stored).

There are three float-float kernels with the same arithmetic. `float-float`
runs one element per thread with the work arrays in thread-local memory, as
`BK1` does. `ff-teams` (`SumFactorizationFFTeams`) runs one element per
team: the hi and lo work arrays are declared at team scope, so they are
shared by the team (shared memory on a GPU), the points of each step are
spread over the team's threads with `parallel for collapse(3)`, and the end
of each parallel region is the barrier between steps. The team size is a
constexpr of nq (`ff_team_size`: nq^3 rounded up to whole warps, 1024 at
nq = 10). `ff-plane` (`SumFactorizationFFPlane`) is the layout of the
production codes: a plane of nq x nq threads per element, each thread
owning the column through its point in registers (hi and lo of nq values),
five phases with a barrier between them, the contraction index as the outer
loop so that every input value and basis entry is loaded once per column,
and the basis read as a warp broadcast. The plane kernel runs twice, with
one element per team and with a batch of elements per team (`ff-plane xN`),
which fills the team at low order; the default batch fills about 128
threads (`ff_plane_batch`: 14 at nq = 3, 8 at nq = 4, 1 from nq = 9) and
`-DCMAKE_CXX_FLAGS=-DFF_PLANE_BATCH=n` at configure time forces one batch for every order.
All float-float kernels sum every output in the same index order and give
the same result bit for bit. On the host the team kernels open parallel
regions per team and are slow; they are meant for the GPU.

The driver runs double, float and the float-float kernels on the same
data, prints one row per kernel and the relative L2 error of float (~1e-7)
and of each float-float kernel (~3e-15) against the double result, all
reduced in double. The last line is the check of the float-float results,
`ok` or `FAIL` against a tolerance (default 1e-10, `BK_TOL=...`), with exit
status 1 on failure:

```sh
build/host/BK1_ff <p> [nelmt] [ntests]   # same arguments and constant data as BK1; the float row prints BK1's norm
BK_RANDOM=1 build/host/BK1_ff 2          # pseudo-random data (the same as BK1_amx's)
for p in 1 2 3 4 5 6 7 8; do build/host/BK1_ff $p 4096 1 | tail -1; done   # the check per order
```

The error-free transformations need strict IEEE semantics: the header
refuses `-ffast-math`, anything subtler shows up as `FAIL`, and the build
selects the native instruction set so `std::fma` is an instruction (the
first output line says `fma = hardware`).

## Intel GPUs (icpx, Arc A770)

The CMake preset `a770` is the A770 workstation's tree: both backends in
one tree, ahead of time for the ocloc device `acm-g10` (Arc A770/A750;
`GPU_DEVICE=...` for others, `ocloc compile --help` lists the names), tinytc
on, compiles one at a time, and no GPU program in
`all`. All GPU builds use
`-fp-model=precise` against icpx's fast default. `scripts/build.sh` sets up
the oneAPI environment (`common.sh`, including the `umf` library the Level
Zero adapter needs), configures `build/a770` and builds the named targets
one after another.

- **One order per binary,** so the device compiler sees one kernel set at a
  time, named by the polynomial order p (`BK5_3`, `bk1_offload_4`): the
  `openmp/` drivers are built with `-DBK_ORDER=<p>`, the other programs with
  `-DNQ=<nq>`. `all_<prog>` builds every order of a program.
- **Memory cap:** every run (`scripts/run.sh`) goes through a systemd user
  scope capped at `MEM_MAX` (8G) that prints its peak memory, and GPU
  compiles never run two at a time. The host data grow as nelmt * nq^3: BK3 at the
  default 524288 elements needs ~7.5 GB at p = 6 and ~11 GB at p = 7.
- **Checks:** the kernel-variant harnesses compare every result with a
  double-precision reference computed on the host, one element per OpenMP
  thread (`BK_PARALLEL=0`: one thread).

```sh
scripts/build.sh BK1_4                     # -> build/a770/BK1_4 (p = 4)
scripts/run.sh BK1_4 4 524288 10           # OMP_TARGET_OFFLOAD=MANDATORY on level_zero:0
PROFILE=1 scripts/run.sh BK1_4 4           # plus the per-kernel device times
scripts/build.sh BK1_ff_4 && scripts/run.sh BK1_ff_4 4 524288 3
scripts/build.sh bk1_offload_4 bk5_sycl_6  # kernel variants: OpenMP offload and SYCL
scripts/build.sh bk3_offload_4 bk3_sycl_4  # BK3: port of the CUDA kernel (cu) and column sweeps (sf)
scripts/build.sh bk_tinytc                 # tinytc driver (preset a770: TINYTC_ROOT=/opt/tinytc/latest)
scripts/study/sweep.sh gpu_results.csv     # BK1, BK3, BK5 at ORDERS="2 4 6", plus STREAM
```

Builds with other flags get their own tree, which `run.sh` and the study
scripts then use too: `BUILD_DIR=build/a770-grf EXTRA_FLAGS=... scripts/build.sh ...`.

`run.sh` shares the GPU with no one: it waits until no other process (other
than display clients) has held it for `GPU_IDLE_SECS` (10) seconds and
repeats a run during which another one appeared (`GPU_GUARD=0` turns this
off).

The A770 has no fp64 hardware. BK1_ff is built and run with the driver's
FP64 emulation (`IGC_EnableDPEmulation=1`, plus
`OverrideDefaultFP64Settings=1` at run time), which the double reference and
the float-float split/join need.

`roofline/stream_omp.cpp` (and `stream_sycl.cpp`, the same test in SYCL)
is a STREAM-like bandwidth test on device-allocated
float arrays: read, write, copy, scale, add and triad, plus `bk1p2`, which
reproduces BK1's p = 2 traffic with one element per thread, and `read9`, 9
reads per write as in BK3 (428 GB/s on the A770: read-heavy traffic does
not reach the 459 GB/s of copy).

### Finding the best kernels

One script per BK runs every kernel variant of a build tree at a fixed
size, `REPS` times, and prints the fastest variants per order and API
(median GDoF/s); the raw runs go to the CSV. They run the binaries of
`BUILD_DIR` (default `build/a770`) on `ONEAPI_DEVICE_SELECTOR` (default
`level_zero:0`, `sycl-ls` lists the devices) and skip the programs a tree
does not have, so they work on a `gpu-sycl` or `gpu-openmp` tree alike:

```sh
export BUILD_DIR=build/gpu-sycl ORDERS="2 4 6" DOFS=1e7 REPS=3
scripts/study/bk1_pick.sh bk1.csv         # BK1: top 5 per order and API
scripts/study/bk3_pick.sh bk3.csv         # BK3: best per order, against the bandwidth roofs
scripts/study/bk5_pick.sh bk5.csv         # BK5: best per order and tensor storage, KERNELS for bk5_sweep.sh
```

The pick is by GDoF/s alone. `bk3_pick.sh` also prints each winner's GB/s
as a fraction of two bandwidth roofs, measured with the tree's `stream_omp`
or `stream_sycl` (or given as `COPY_GBS=... READ9_GBS=...`), to show how
close to the memory bound it is.

A variant the device cannot launch, such as a work-group above its limit
(512 work-items on the UHD 750, 1024 on the A770), prints `skipped` and the
rest still run. `BK_KERNEL="-name ..."` leaves variants out; the naive BK1
baselines (built only with `BK_GPU_BASELINES=ON`) can run long enough at
high order to trip a small GPU's hang check and lose the device.
One tree can serve several GPUs (`BK_GPU_DEVICE="acm-g10,rkl"`), e.g. on
this machine the A770 and the UHD 750 of the Core i7-11700K:

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:1 scripts/study/bk5_pick.sh bk5_uhd.csv
```

### Results

`results/a770/` holds the measurements of the kernel variants (OpenMP
offload, SYCL, tinytc) with the scripts that produced them; the roofline
(`roofline_a770.png`) and `tinytc_compare_a770.txt` summarize them. At
~5.6e7 DoFs the best OpenMP or SYCL variant reaches, in GDoF/s:

| p                | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
|------------------|------|------|------|------|------|------|------|------|
| BK1              | 19.8 | 24.4 | 23.4 | 25.5 | 26.3 | 29.5 | 27.6 | 26.0 |
| BK5 (6 factors)  | 14.0 | 13.5 | 13.5 | 12.2 | 12.3 | 11.7 | 13.5 | 11.5 |

All of them are memory-bound, at 71-97% of the STREAM bandwidth roof.

BK3 (`bk3_offload`, `bk3_sycl`; `results/a770/bk3_pick_a770.txt`, median of
3 runs at ~2e7 DoFs, best of the two APIs) against two roofs measured by
`stream_omp` in the same run: STREAM copy (454 GB/s) and read9 (426 GB/s),
which has BK3's mix of ~90% reads (in + G against out):

| p             | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
|---------------|------|------|------|------|------|------|------|------|
| GDoF/s        | 4.48 | 6.30 | 6.84 | 7.60 | 7.40 | 8.90 | 7.51 | 8.25 |
| % of copy     | 88   | 90   | 83   | 83   | 75   | 86   | 70   | 74   |
| % of read9    | 93   | 96   | 88   | 88   | 80   | 91   | 74   | 79   |

What mattered: SIMD8 (`_sg8` in SYCL; IGC picks it by itself for the OpenMP
`cu`) and the basis matrices held in registers (variant suffix `r`).
At p = 5, 7, 8 the compute steps (~11 GDoF/s alone) and the G stream (~9.4
GDoF/s alone) overlap poorly; SLM padding, fill groups, prefetching G and
a compile-time work-group size changed little there.

Against the Kokkos CEED_BK kernels of
[dealii-X/benchmarks](https://github.com/dealii-X/benchmarks) (Kokkos 5.3.0,
SYCL ahead of time for DG2, their drivers switched to float;
`scripts/study/kokkos_build.sh`, then `kokkos_compare.sh`) at the same
sizes, with Kokkos at its fastest element batch, ours are 1.30-2.16x faster
on BK1 and BK3 and 1.10-2.10x on BK5 (`results/a770/kokkos_compare_a770.png`,
`.txt`).

BK3 in tinytc (`tinytc/ir/bk3_blk.ir`, `bk3_ec.ir`: BK1's interpolation
around BK5's Laplacian, a work-group per block of 8-32 elements, the
sub-group size 8 or 16 as `TT_SG`) keeps U, R and S of 3 B nq^3 floats in
local memory up to p = 6 (B = 8). At p = 7 and 8 these exceed 64 KiB, and
`bk3_blk_gs.ir`, `bk3_ec_gs.ir` keep S in global memory instead (a block of
B nq^3 floats per work-group; `TT_SPILL`, on by default only there; about
10% slower where it isn't needed, 8.0 vs 8.7 GDoF/s at p = 6). With the best
configuration per order (`tinytc_survey_bk3_a770.csv`), in its blocked
layout it matches or slightly beats our best kernel up to p = 6
(1.00-1.09x) and reaches 0.92x and 0.88x at p = 7 and 8; in our
element-contiguous layout it reaches 0.62-0.80x of it
(`bk3_tinytc_compare_a770.png`, `.txt`; `scripts/study/bk3_tinytc_compare.sh`).

The measurements below are of the portable drivers in `openmp/` (one
element per thread), on an Arc A770 (16 GB) with icpx 2026.1, compute
runtime 26.22, 524288 elements, in float. Each value is the minimum
over 10 runs (3 for BK1_ff at p = 4, 6). Separate runs differ by up to
~20% (BK1 at p = 2 measured 12.7-16.0 GDoF/s); STREAM varies by under 1%.

| STREAM | read | write | copy | scale | add | triad | bk1p2 |
|--------|------|-------|------|-------|-----|-------|-------|
| GB/s   | 354  | 430   | 458  | 457   | 444 | 438   | 118   |

| GDoF/s (GB/s) | BK1          | BK3          | BK5 (nq = p+1) | BK1_ff float-float, best kernel |
|---------------|--------------|--------------|----------------|---------------------------------|
| p = 2         | 13.4 (235)   | 0.18 (11.7)  | 6.77 (217)     | 2.04 (ff-plane x8)              |
| p = 4         | 0.099 (1.5)  | 0.040 (2.0)  | 0.34 (10.8)    | 1.36 (ff-teams)                 |
| p = 6         | 0.081 (1.1)  | 0.035 (1.5)  | 0.31 (9.8)     | 1.17 (ff-teams)                 |

At p = 2, BK1 and BK5 run at about half of STREAM copy. From p = 4, the
one-element-per-thread kernels with a per-thread `2 * nq^3` work array fall
by 20-135x. BK3 already falls at p = 2. The float-float `ff-teams` kernel,
which keeps its work arrays in team-shared memory, then runs ~8x faster than
the float BK1 kernel (likely register spills of the per-thread array, not
confirmed). Emulated double runs at 0.05-0.09 GDoF/s. All BK1_ff checks pass
on the A770 at p = 2, 4 and 6, with and without `BK_RANDOM=1`.

## C for Metal versions of BK1 and BK5

`cm/` has BK1 and BK5 (G6) in Intel C for Metal, an explicit-SIMD language
for Intel GPUs, with SYCL hosts. They are built by `make -C cm` (the 2021 CM
SDK and `ocloc`), not by CMake. On the A770 they reach 85-98% (BK1) and
83-96% (BK5) of a CM copy kernel's bandwidth, ahead of the best OpenMP/SYCL
variant at p = 3-5 and 7-8 (BK1) and at p = 4-6 and 8 (BK5).
[`cm/README.md`](cm/README.md) explains C for Metal, where to get it, and how
to build and run the kernels.

## Apple AMX version of BK1

`amx/BK1_amx.cpp` runs the BK1 sum factorization on the Apple AMX
coprocessor; elsewhere `amx/amx.h` emulates the AMX instructions in
software, so the kernel logic can be checked on any machine. The same binary
also runs the CPU baselines (`BK_KERNEL=neon`: 4-wide NEON FMAs across the
elements of a batch; `BK_KERNEL=refv`: the reference with unit-stride inner
loops). The header of `amx/BK1_amx.cpp` lists all environment variables.

```sh
cmake --preset host && cmake --build build/host   # accel_gemm only on macOS (Accelerate)
build/host/BK1_amx <p> [nelmt] [ntests]   # same arguments as BK1, p = 1..14
BK_KERNEL=neon BK_LAYOUT=soa build/host/BK1_amx 4

amx/scripts/validate.sh                   # every kernel against the reference; exit 1 on failure
amx/scripts/throughput.sh results.csv     # sweep (all kernels from BK1_amx): kernel,p,target,nelmt,dofs,gdofs,gbs
amx/scripts/high_order.sh results_high.csv   # the same for p = 9..14 (AMX binary only)
OUT=tune.csv amx/scripts/tune.sh          # best layout/path/batch per order
build/host/bw_test    > bw.csv            # bandwidth and FMA roofs: test,threads,bytes,seconds,GB_per_s
build/host/amx_pipe   > pipe.txt          # AMX load-to-use and store behaviour (one unit)
build/host/accel_gemm > gemm.csv          # Accelerate sgemm calibration (macOS)

amx/scripts/plot_throughput.py results.csv throughput.png   # per kernel and order, AMX/NEON speedup
amx/scripts/plot_roofline.py results.csv bw.csv roofline.png # plateaus against the roofs
```

The scripts find `BK1` and `BK1_amx` in `build/host` and run from
any directory; set `BK1_AMX=...` or `BK1=...` to use other builds. The
plots need matplotlib.
