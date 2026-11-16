# BK1 and BK5 in C for Metal

This directory holds BK1 and BK5 (symmetric factors, G6) written in Intel C for
Metal, for Intel GPUs, with SYCL host drivers. They are built by the
`Makefile` here, not by CMake.

## What C for Metal is

C for Metal (CM) is Intel's explicit-SIMD language for its GPUs. "Metal" means
close to the metal; it has nothing to do with Apple's Metal API.

- **Language:** a C++ subset plus `vector<T, N>` and `matrix<T, R, C>` types
  that live in registers, `select`/`replicate`/`format` region operations that
  map to the hardware's register regioning, and intrinsics for the memory
  messages (block, scattered and SLM reads and writes) and barriers.
- **Execution model:** a CM kernel describes one hardware thread, not one
  work-item. The programmer picks the SIMD width of each operation and which
  registers hold which data, much like AVX intrinsics on a CPU, while SYCL and
  OpenCL C leave both to the compiler.
- **Toolchain:** the offline compiler `cmc` (clang based, package
  `intel-igc-cm`) turns a kernel into SPIR-V with the vector-compute
  extension; `ocloc` with `-vc-codegen` then compiles that to a device binary
  through IGC's vector-compute backend. A host
  program loads the binary through Level Zero, OpenCL or SYCL, with the SIMD1
  dispatch model: every range counts hardware threads.
- **Status:** Intel's current form of the same model is SYCL ESIMD, inside
  oneAPI. CM itself is still maintained, and `cmc` ships with Intel's GPU
  driver packages, not with oneAPI.

## Where to find it

| What | Where |
|---|---|
| Compiler sources and language specification | <https://github.com/intel/cm-compiler> (spec: `clang/compute_sdk/docs/cmlangspec/`) |
| `cmc`, prebuilt | package `intel-igc-cm` in Intel's GPU package repository (<https://repositories.intel.com/gpu/ubuntu>, release lists on <https://dgpu-docs.intel.com>); used here: 1.0.225.54093 (clang 11) |
| `ocloc` and the GPU driver | <https://github.com/intel/compute-runtime>, or packages `intel-ocloc`, `intel-opencl-icd`, `libze-intel-gpu1` of the same repository |
| Background | "C-for-Metal: High Performance SIMD Programming on Intel GPUs", CGO 2021, <https://arxiv.org/abs/2101.11049> |

A `cmc` writes SPIR-V that only the IGC of its own driver release reads: the
SPIR-V of `intel-igc-cm` 1.0.225 is rejected by newer IGCs ("verify Module
failed"). The kernels are therefore compiled to device binaries by the `ocloc`
of that release (driver build 1146: compute-runtime 25.18, IGC 2.11), and
those binaries run on the system's newer driver. `install_toolchain.sh`
installs this pair in user space, without touching the system driver:

```sh
cm/install_toolchain.sh            # -> ~/opt/intel-cm-1146/bin/{cmc,ocloc-cm} (about 100 MB of packages)
```

It unpacks `intel-igc-cm`, `libigc2`, `libigdfcl2` and `intel-ocloc` of that
release and writes two wrappers that give only `cmc` and `ocloc-cm` the
matched libraries. A system-wide install of the same packages works too; then
set `CMC` and `OCLOC` (or `CM_TOOLS`) for the Makefile.

**Toolchain versions and other GPUs.** The kernels are tested with
`intel-igc-cm` 1.0.225.54093 and the `ocloc` of driver build 1146, on the
Arc A770, with the binaries run on the system's newer driver
(compute-runtime 26.40). The Makefile does not check which toolchain it is
given.

- `cmc` and `ocloc` must come from the same driver release. For a newer
  `intel-igc-cm`, `install_toolchain.sh -b <build> -c <version>` installs
  that release's pair; the package lists on <https://dgpu-docs.intel.com>
  show which build goes with which version.
- An `ocloc` from another release rejects the SPIR-V with
  `spirv_read_verify: verify Module failed`. That error means a mismatched
  pair, not a problem in the kernels.
- Another GPU: set `GPU_DEVICE` to a name both `cmc -march` and
  `ocloc -device` accept (e.g. `bmg` for Battlemage; `ocloc compile --help`
  lists them), and use a compiler release recent enough to know it. Only the
  A770 is tested, and the tuning (row padding, elements per thread,
  work-group sizes, the large register file from nq = 6) was done there.

## Building

Requirements: `cmc` and its matched `ocloc` (`CM_TOOLS`, default
`~/opt/intel-cm-1146/bin`, see above) and oneAPI (`ONEAPI_ROOT`, default
`/opt/intel/oneapi`) for `icpx`. The oneAPI environment does not need to be sourced, for the build
or to run the hosts.

```sh
make -C cm                         # everything -> build/cm/
make -C cm bk5 ORDERS="3 4"        # one kernel, some orders
make -C cm KEEP=1 -B bk1 ORDERS=7  # keep the Gen assembly in build/cm/dump_bk1_cm_7/
make -C cm check                   # run both hosts at every order (correctness)
```

Each kernel is compiled once per order p, with `-march=$(GPU_DEVICE)` (default
`acm-g10`, the Arc A770/A750) and `-DNQ=<nq>` (BK1: nq = p + 2, BK5:
nq = p + 1), into `build/cm/<kernel>_cm_<p>.bin`, and the build prints
the register spill, if any, of each binary. From nq = 6 a kernel uses the
large register file (`-double-GRF`: 256 registers per thread, half the threads
per EU). `<kernel>_cm_<p>_e<n>.bin` runs n elements per thread, the next one's
loads issued before the current one is computed; `EPT_BUILDS` lists the ones
that are the best configuration of their order (BK1 at p = 1 and 3, BK5 at
p = 1 and 2). The hosts `build/cm/bk1_cm` and `build/cm/bk5_cm` are built with
`icpx -fsycl -fp-model=precise`. The `Makefile` header lists every variable
(`GPU_DEVICE`, `GRF`, `CMC_FLAGS`, ...).

## Running

```sh
build/cm/bk1_cm <p> [nelmt] [ntests]
build/cm/bk1_cm 4 448000 20
BK_EPT=2 build/cm/bk1_cm 3 875000 20   # the _e2 binary
build/cm/bk5_cm 8 76817 20
BK_WG="32 64" build/cm/bk5_cm 5        # work-group sizes (threads per group)
```

A host first measures the bandwidth of `copy_cm`, a block-copy kernel moving
as many bytes as the benchmark, then runs the kernel at each work-group size
of `BK_WG`. It prints host time (submit to completion) and device time (event
profiling) in GDoF/s and GB/s, the device GB/s as a share of `copy_cm`'s, and
the error against the double-precision reference of `bk_harness.h` (exit
status 1 on a failure). `BK_CM_DIR` points to another directory of binaries;
`ONEAPI_DEVICE_SELECTOR=opencl:gpu` uses the OpenCL backend instead of Level
Zero; `CM_DEVICE=<name>` prefers a GPU whose name contains it.

## The kernels

**BK1** (`bk1_cm_genx.cpp`): each thread runs a whole element in registers.
Each of the six contractions sums over the slowest index of a register tensor
and appends the new index as the fastest, so it is a small GEMM per element.

- p = 1..6: every tensor is a `rows x P` register matrix with its fast index
  padded to P = 4 or 8, and every instruction is a full SIMD16 `mad`.
- p = 7, 8: padding 9 or 10 to 16 does not fit the register file, so an index
  of extent 9 or 10 is stored as a main block of 8 and a tail of 1 or 2. Main
  blocks take one SIMD8 `mad` per row, tails one SIMD16 `mad` per 16 rows of a
  column. At p = 8 one element still does not fit 256 registers, so a pair of
  threads shares it: each computes half of the quadrature index and a partial
  result, and one hands its partial to the other through shared local memory.

**BK5** (`bk5_cm_genx.cpp`): each plane of the element is held flat (memory
order) for the loads and the pointwise r steps; the s and t derivatives, the
fluxes and their back-contractions work on rows padded to P = 4 or 8, so that
every register write is aligned. G is read in layout 1 (`G(e, i, j, f, k)`),
one contiguous run per plane, and the output is packed per element and written
by aligned block writes. A build without arithmetic runs at the same speed:
the kernel is limited by its memory access pattern. At p = 8 (nq = 9) one
element does not fit a thread's registers; as in the SYCL kernel, a work-group
of nine threads shares it through shared local memory, each thread owning one
plane, with two barriers per element.

The other files: `cm_common.h` (memory and SLM helpers, `copy_cm`),
`cm_sycl.hpp` (loading the binaries through Level Zero or OpenCL interop,
launching, timing, reporting), `bk1_cm.cpp` and `bk5_cm.cpp` (the hosts).

## Results on the Arc A770

Host GDoF/s at ~5.6e7 DoFs, best configuration per order, against the best
OpenMP or SYCL variant of the main README. Details and the variants tried:
`results/a770/cm_bk1_a770.txt`, `results/a770/cm_bk5_a770.txt`.

| p | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| BK1, CM | 17.6 | 21.9 | 26.9 | 27.1 | 29.8 | 28.3 | 30.7 | 30.2 |
| BK1, best OpenMP/SYCL | 19.8 | 24.4 | 23.4 | 25.5 | 26.3 | 29.5 | 27.6 | 26.0 |
| BK5, CM | 12.8 | 11.6 | 13.3 | 12.3 | 12.8 | 12.6 | 12.7 | 12.3 |
| BK5, % of copy_cm | 93 | 84 | 96 | 88 | 91 | 89 | 90 | 87 |
| BK5, best OpenMP/SYCL | 14.0 | 13.5 | 13.5 | 12.2 | 12.3 | 11.7 | 13.5 | 11.5 |

BK1 reaches 86-97% of `copy_cm`'s bandwidth (device time) and 5-21% of the
FP32 peak; BK5 is memory-bound (below 2% of the FP32 peak).
