# BK1, BK3 and BK5 in the Tiny Tensor Compiler

The [Tiny Tensor Compiler](https://github.com/intel/tiny-tensor-compiler)
(tinytc) is Intel's experimental compiler for small tensor contractions on
Intel GPUs. Kernels are written in its MLIR-like tensor language (`.ir`
files), and the host program compiles them to SPIR-V at run time and
launches them through SYCL. This folder has BK1, BK3 and BK5 written that
way, a driver that checks them against this repository's references, and
what we found doing it. Nothing else in the repository needs tinytc.

`ir/bk1.ir` and `ir/bk5.ir` are the kernels of the tinytc author in
[dealii-X/benchmarks](https://github.com/dealii-X/benchmarks/tree/opencl/alternatives)
(branch `opencl`, `alternatives/src/`), unchanged. The rest is ours.

## Getting tinytc

tinytc is built from source; the kernels here are written for v0.5.0. It
needs CMake, icpx (oneAPI), re2c 3.0 or later and bison 3.8.2 or later:

```sh
source /opt/intel/oneapi/setvars.sh
git clone -b v0.5.0 https://github.com/intel/tiny-tensor-compiler.git
cmake -S tiny-tensor-compiler -B tinytc-build -G Ninja -DCMAKE_CXX_COMPILER=icpx \
      -DBUILD_SHARED_LIBS=YES -DCMAKE_INSTALL_PREFIX=<prefix>
cmake --build tinytc-build && cmake --install tinytc-build
export TINYTC_ROOT=<prefix>
```

The install has no environment script. `. scripts/tinytc_env.sh` (bash)
puts it and the oneAPI runtime into the current shell, for the `tinytc`
and `tinytc-opt` tools: a tinytc built with icpx links oneAPI libraries, so
even the tools need oneAPI. `TINYTC_ROOT` defaults to `/opt/tinytc/latest`,
where the A770 machine has it.

tinytc 0.5 has no Arc A-series (DG2, Xe-HPG) target: `tinytc -d` rejects
`acm-g10`, `dg2` and `xe_hpg`. The driver compiles for the device it runs
on (`create_core_info(device)`), which works on the A770; for offline
checks of a kernel, `-d tgl` has the same sub-group sizes (8, 16, 32).

## Building and running

```sh
cmake --preset gpu-sycl -DBK_ENABLE_TINYTC=ON    # TINYTC_ROOT from the environment, or -DTINYTC_ROOT=...
cmake --build build/gpu-sycl --target bk_tinytc
build/gpu-sycl/bk_tinytc <1|3|5> <p> [nelmt] [ntests]
```

On the A770 machine, `scripts/build.sh bk_tinytc` builds it into
`build/a770` (the `a770` preset has tinytc on), and `scripts/run.sh
bk_tinytc ...` runs it. The kernels are read from `tinytc/ir/` of the
source tree at run time (`TT_IR_DIR` overrides). Every run prints GDoF/s
and GB/s and the error against the double-precision reference.

The main settings, as environment variables (the comment at the top of
`bk_tinytc.cpp` lists them all):

| Variable | Default | Meaning |
|---|---|---|
| `TT_LAYOUT` | `ec` | `ec`: kernels on our element-contiguous arrays; `blk`: tinytc's blocked layout (B elements, element innermost), converted by the driver; `blk_ch` (BK3): `blk` with the r direction chained in registers (`bk3_blk_ch.ir`) |
| `TT_B` | 16 | elements per block, one block per work-group |
| `TT_SG` | 16 | BK3: sub-group size, the rows of each cooperative-matrix tile (8 or 16) |
| `TT_WGS` | by nq | work-group size |
| `TT_SPILL` | where needed | BK3: keep one work array in global memory (`*_gs.ir`) |

## The kernels

All of them run a block of B elements per work-group, with the element
index as the rows of each cooperative-matrix tile, and each contraction
step as one `foreach_tile` loop over the work-group's sub-groups.

| File | Kernel |
|---|---|
| `bk1.ir`, `bk5.ir` | the original BK1 and BK5 (blocked layout) |
| `bk5_blk.ir` | `bk5.ir` with a race fixed: the steps that read-modify-write `out` in global memory need `barrier.global.local` before them, not `barrier.local` (wrong results at large work-groups otherwise) |
| `bk1_ec.ir`, `bk5_ec.ir` | the same on our element-contiguous arrays |
| `bk3_blk.ir`, `bk3_ec.ir` | BK3: BK1's interpolation around BK5's Laplacian, in either layout; three work arrays of B nq^3 floats in local memory |
| `bk3_blk_gs.ir`, `bk3_ec_gs.ir` | BK3 at p = 7, 8, where those arrays exceed 64 KiB: one of them in global memory |
| `bk3_blk_ch.ir` | BK3 with the derivatives along p and q taken before the interpolation along r, so one tile runs the whole r direction (interpolation, three derivatives, G, and back) in registers; the work arrays shrink to B nm nq^2 floats |

## Measuring

```sh
scripts/study/tinytc_survey.sh survey.csv          # orders, layouts (BK3: also blk_ch), work-group, sub-group and block sizes
scripts/study/tinytc_final.sh picks.txt final.csv  # chosen configurations, several runs each
scripts/study/bk3_tinytc_compare.sh picks.txt bk3.csv   # BK3: tinytc against our OpenMP and SYCL kernels
```

## Results on the Arc A770

GDoF/s, icpx 2026.1, float, tinytc 0.5.0, best configuration per order
(`results/a770/`). BK1 and BK5 at ~5.6e7 DoFs (`tinytc_compare_a770.txt`):
in the blocked layout tinytc is ahead of our best OpenMP or SYCL variant
up to p = 6 (BK1 1.06-1.22x, BK5 1.01-1.11x; BK5 at p = 7 0.88x), and in
our element-contiguous layout at 0.51-0.71x. With B = 16 the work arrays
no longer fit in local memory for BK1 at p = 7, 8 and BK5 at p = 8.

BK3 at 2e7 DoFs, median of 3 runs (`bk3_tinytc_compare_a770.txt`):

| p | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ours (OpenMP or SYCL) | 4.33 | 5.95 | 6.44 | 7.28 | 6.91 | 8.17 | 7.27 | 8.08 |
| tinytc, blocked | 4.55 | 6.00 | 6.99 | 7.30 | 7.54 | 8.37 | 6.68 | 7.10 |
| tinytc, element-contiguous | 3.48 | 4.75 | 4.66 | 5.35 | 5.03 | 5.60 | 4.80 | 5.05 |

At p = 7, 8 the BK3 kernels run with one array in global memory (`_gs`),
which costs about 10% where it isn't needed (8.0 against 8.7 at p = 6).
The layout matters more than any setting: on our arrays the same kernels
reach 0.62-0.80x. Sub-group size 8 with B = 8 was fastest from p = 4.

The chained kernel (`blk_ch`), against `blk` in the same session (median of
3 alternating runs at 2e7 DoFs, configurations from a survey of an earlier
version of it):

| p | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| `blk_ch` | 4.63 | 6.26 | 6.84 | 7.87 | 8.01 | 8.79 | 7.50 |
| `blk` (p = 7: `blk_gs`) | 4.65 | 6.22 | 7.56 | 7.54 | 7.76 | 8.71 | 6.96 |

Its smaller arrays fit in local memory at p = 7, where it gains 8%; at
p = 8 they still do not (no `_gs` version yet). The fused tile costs more
arithmetic than the passes it replaces, and f32 has no matrix engine on
DG2: only with B D precomputed on the host (one contraction instead of two
each way) does it come out even or ahead, and before that it lost up to
22%.

## Upstream

`upstream_issues.md` drafts three reports for intel/tiny-tensor-compiler
(not posted): a DG2 target, local memory over 64 KiB failing with an
undiagnosed `tinytc error 18` (reproducer in `repro_slm/`), and
`insert-barrier` not seeing the local-memory accesses of `foreach_tile`
loops, which therefore need their barriers placed by hand.
