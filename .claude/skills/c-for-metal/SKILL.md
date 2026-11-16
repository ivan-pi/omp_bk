---
name: c-for-metal
description: Write, compile, run, inspect and debug Intel C for Metal (CM) GPU kernels (explicit-SIMD C++ with vector/matrix register types, select regions and genx intrinsics). Compile with cmc to LLVM IR / SPIR-V, finalize with ocloc -vc-codegen (vISA, Gen asm), and launch from OpenCL, Level Zero or SYCL (kernel_compiler SPIR-V). Also covers the legacy CM runtime host API (CmDevice, CmProgram, CmKernel, CmThreadSpace, CmTask, CmQueue, libigfxcmrt, CPU emulation), deciding whether explicit SIMD (CM, like AVX intrinsics on a GPU) or SIMT (CUDA/SYCL) fits a problem, and porting between CM and SYCL ESIMD (simd types, xmx::dpas, invoke_simd), which is the recommended path for new code. Use it whenever the user mentions C for Metal, C for Media, CM kernels, cmc, _GENX_MAIN_, SurfaceIndex, CmDevice, vector_ref/matrix_ref, llvm.genx intrinsics, vISA, -vc-codegen, VectorComputeINTEL SPIR-V, the CM SDK, ESIMD, or register-level explicit-SIMD programming of Intel Gen/Xe GPUs (Arc, DG2, PVC), even if they don't say "C for Metal".
---

# C for Metal (CM) development

CM is Intel's explicit-SIMD GPU language: a C++ subset plus `vector<T,N>` and
`matrix<T,R,C>` types that live in registers, region-select operations that
map directly to Gen/Xe register regioning, and intrinsics that mirror the ISA
(block/2D/scattered memory messages, atomics, SLM, dpas). **A CM kernel
describes one hardware thread, not one work-item.** You choose the SIMD
width of each operation, which registers hold which data, and how lanes
exchange data, so the compiler has no auto-vectorization or divergence
analysis to get wrong. The CGO 2021 paper (arXiv:2101.11049) reports
1.1-2.7x over tuned OpenCL.

**Scope of this skill:** CM with the current Linux packages
(`intel-igc-cm` from Intel's GPU repository, installed with the driver
release it belongs to), for existing CM code and for systems that stay on
CM. Older releases (the 2021 CM SDK, the 2022 GitHub release) are in
**`references/legacy-sdk.md`**; read it only for code or systems tied to
them, the legacy CM runtime's `.isa`, or the CPU emulator. **For new
development, recommend SYCL ESIMD** (`references/esimd.md`). It is the
same programming model inside current oneAPI compilers, with `simd<T,N>`,
`select`, `xmx::dpas` and `invoke_simd` instead of a separate CM toolchain,
and it runs on the same IGC vector-compute backend.

## Terms

| Term | Meaning |
|---|---|
| Gen9 … Gen12LP, Xe-HPG, Xe2 | Intel GPU generations. Gen12LP: Tiger/Rocket/Alder Lake integrated GPUs; Xe-HPG: **DG2** (Arc A-series, "Alchemist"); Xe-HPC: PVC (data-center Max); Xe2: Battlemage, Lunar Lake |
| **IP version** | `major.minor.revision` hardware id (e.g. 12.55.8 for DG2-G10, 12.1.0 for Rocket Lake): `-march`, `ocloc -device`, `CM_GENX` (`references/targets.md`) |
| hardware thread, **SIMD** / **SIMT** | a CM kernel instance is one hardware thread issuing SIMD instructions (explicit SIMD); CUDA/SYCL program one lane each (SIMT). **SIMD1 dispatch**: the NDRange counts threads |
| **GRF** | general register file: 128 (or 256, "large GRF" / `-double-GRF`) 32-byte registers per thread. CM `vector`/`matrix` objects live there |
| **SLM** | shared local memory, per work-group (OpenCL `local`, CUDA `__shared__`) |
| surface / `buffer_t` (stateful) vs **SVM** / **USM** (stateless) | a bounds-checked buffer handle (`SurfaceIndex`) vs a raw 64-bit pointer: OpenCL **S**hared **V**irtual **M**emory, SYCL/Level Zero **U**nified **S**hared **M**emory |
| message, data port, **OWord** | memory accesses are `send` messages to the data port; legacy block messages move 1-8 **OWords** (16 B each, ≤ 128 B) |
| **LSC** | **L**oad/**S**tore **C**ache data port of DG2 and newer; its API is `cm_load`/`cm_store`/`cm_atomic` (byte offsets, ≤ 256 B blocks) |
| **XMX**, **DPAS** | Xe Matrix eXtensions (systolic arrays) and their **D**ot-**P**roduct-**A**ccumulate-**S**ystolic instruction (`cm_dpas`) |
| **IGC**, VC, **vISA** | Intel Graphics Compiler (the driver's backend); its **V**ector-**C**ompute path for CM/ESIMD (`-vc-codegen`); vISA, IGC's virtual ISA before Gen assembly |
| **NEO**, **ocloc**, **L0** | Intel's GPU compute runtime (OpenCL + Level Zero drivers), its offline compiler, Level Zero |
| **SPIR-V**, zebin, JIT / AOT | Khronos IR (CM uses `SPV_INTEL_vector_compute`); the ELF device-binary format; finalize at run time vs ahead of time |
| **ESIMD** | SYCL's Explicit SIMD extension, CM's successor inside oneAPI |
| `_GENX_MAIN_`, `_GENX_` | `__declspec(genx_main)`: a kernel entry point; `__declspec(genx)`: optional device-function marker (`inline` decides inlining). "GenX" = Intel's "Gen" GPU target (`references/language.md` §8) |
| **CLHPP** | the OpenCL C++ bindings (`CL/opencl.hpp`) used by the examples |
| `ocloc-cm`, `cm_run` | **not Intel tools**: wrappers made by this skill's `scripts/install_cm_toolchain.sh` |

## Explicit SIMD vs. SIMT: how to think about CM

CM is the GPU counterpart of CPU vector programming. If you know AVX/AVX-512
intrinsics, the GNU vector extension (`typedef float v8f __attribute__((vector_size(32)))`)
or `std::experimental::simd`, you already know the model: one thread works
on whole vectors, the vector width is in the type, and lanes exchange data
through explicit shuffles. CM's version uses C++ templates (`vector<float,16>`,
`v.select<8,2>(1)`, `m.format<int>()`, `v.merge(x, mask)`) instead of names like
`_mm512_permutexvar_ps`, and its "shuffles" are free register regions.

CUDA and OpenCL C are **implicit SIMD** (SIMT): you write scalar code for one
lane, and the compiler packs 8/16/32 work-items into each hardware thread's
SIMD instructions. The warp or sub-group is a vector you never see, and lanes
communicate through shuffles, sub-group operations or shared memory.

A side-by-side table (CPU SIMD / CM / SIMT) and **how SYCL/OpenCL sub-groups
relate to ESIMD and CM** are in **`references/simt-vs-simd.md`**. In short,
a sub-group *is* the hardware thread that one CM/ESIMD work-item programs
directly: its size is the SIMD width the compiler chose, and each of its
lanes is one CM vector element.

### When CM is the right tool
- You need performance the SIMT compiler won't give: register blocking
  without spills, shuffles done as register regions, the SIMD width varied
  per phase, 2D block loads of whole tiles, or explicit `dpas` (XMX)
  scheduling. Typical workloads are GEMM and convolution tiles, image
  filters, sorting networks, histograms, transposes, and media-style 2D
  processing.
- The work per thread is a natural vector or tile (an 8x32 pixel block, a
  16x16 matrix tile).
- Irregular, mixed serial/parallel code where SIMT divergence or
  under-filled sub-groups waste lanes.
- You are maintaining existing CM kernels, or a system tied to the CM
  runtime/media stack.

### Kernel types that fit
- **FP32 numerics:** dense tiles (GEMM/convolution with register
  blocking), stencils, reductions, transposes, and batched small dense
  solvers held entirely in registers (`examples/gesv_genx.cpp`). FP16/BF16 matrix products
  go through XMX `cm_dpas` (8x8 per instruction on DG2). **DG2 has no FP64
  hardware**, so double-precision kernels are emulated and slow; FP64 belongs
  on PVC or a CPU.
- **Sparse and irregular:** CSR SpMV with one thread per row and a slice
  width (SIMD8/16/32) picked per row (`examples/spmv_genx.cpp`, the paper's
  case study), graph frontiers, gathers with masked tails.
- **Integer and byte work:** histograms (`write_atomic`, or per-thread
  register histograms merged at the end), sorting networks (bitonic sort
  in registers), prefix sums, bit manipulation (`cm_cbit`, `cm_fbl`/`cm_fbh`,
  `cm_bfn`, `cm_bf_extract`).
- **Multi-pass kernels: count the passes.** When every launch streams the whole array, time ≈
  passes x one copy, and the arithmetic hides (measured on an Arc A770: ~0.3 ms per 64 MiB pass). So fuse launches: a
  bitonic sort (2^24 keys, 256 in registers per thread) can do M cross-thread strides in one launch
  by loading 2^M pieces of 256/2^M contiguous keys, which turns them into in-register strides. M = 1
  (one launch per stride, the paper's scheme): 153 launches, 49 ms; M = 4: 57, 18 ms; M = 7: 44,
  14.6 ms (1150 Mkeys/s); M = 8 is slower, because the pieces shrink to 4-byte gathers. Keep pieces
  at 8 B or more and measure.
- **Histograms:** keep each thread's bins in registers, then merge them per group in SLM before the
  global atomics. On an Arc A770, 4096 threads x 256 global atomics capped the merge at ~90 GB/s,
  against ~250-330 GB/s with it; SLM atomics fell to ~21 GB/s on a uniform image.
- **Bandwidth-bound reductions** such as argmax, min/max, sum or count: keep
  per-lane partial results in registers, issue several block reads before
  using them, and give each thread a short chunk (256-512 elements was best
  on DG2: more threads in flight). Finish with a reduction tree over
  structure-of-arrays partials, not a single thread (that took 0.36 ms of a
  1.1 ms run). Measure against a read-only kernel with the same access
  pattern (`examples/argmax_genx.cpp`, `readsum_i32`).
- **Scans (prefix sums):** inside a vector, use the log-step scan on
  `select` regions, where an exclusive scan is the inclusive one shifted by
  one lane. It is free next to memory: the rescan pass ran at copy speed.
  For arrays, reduce-then-scan (`examples/scan_genx.cpp`) costs ~1.5-1.65x
  a copy, and it is the method to use on DG2. The single-pass decoupled
  look-back scan (`scan_lookback`, same file) is correct but 14x slower
  (28.6 ms vs 2.03 ms). On DG2 the legacy `cm_fence` is a GPU-scope fence
  that evicts L1, and each look-back step is an atomic round trip.
- **Images and media:** 2D media-block reads with hardware edge clamping,
  9-tap stencils as register `select`s (`examples/linear_genx.cpp`), format
  conversions with saturation, transposes.
- **Text and bytes:** `vector<uchar, 32...>` compares produce masks for
  character classes; `.any()`/`.all()`, `cm_pack_mask` and `cm_fbl` find the
  first match; byte shuffles are `select`/`iselect`. Useful for scanning,
  tokenizing and pattern matching, with irregular output via atomics.
- **Gen12LP integrated GPUs run all of it** except XMX and LSC: every
  OpenCL/SYCL example passed on a UHD 750, at about 1/10 of an Arc A770's
  throughput (SpMV: 2.06 ms vs 0.20 ms). The legacy CM runtime supports
  Gen12 and older, not DG2.

### When it is not
- Portable or cross-vendor code: CM runs on Intel GPUs only, Gen9 and newer.
- New code on a current oneAPI: use ESIMD, or plain SYCL when the compiler's
  vectorization is good enough.
- Simple element-wise or reduction kernels that SYCL/OpenCL already run at
  bandwidth: CM adds toolchain burden for no gain.
- CPU targets: there is no CM CPU backend, only an archived emulator for
  debugging.
- Teams without GPU-ISA knowledge (GRF size, regions, message types): the
  explicit model makes those your responsibility.

## The layers of CM

| Layer | What it is | Status, docs |
|---|---|---|
| 1. Kernel language | C++ subset + `vector`/`matrix`/refs + intrinsics; `_GENX_MAIN_` kernels; `cm/cmtl.h` template library | `references/language.md`, `memory.md` |
| 2. Offline compiler `cmc` | clang front end (`-march=<device>`) → LLVM IR with `llvm.genx.*` intrinsics → SPIR-V (`-emit-spirv`) or a device binary (zebin). Kernels are normally **compiled ahead of time** | `intel-igc-cm` 1.0.225; `references/pipeline.md` |
| 3. Backend (IGC "VC") | GenX LLVM passes (rdregion/wrregion, baling) → vISA → finalizer → Gen ISA; driven by `-vc-codegen`. Also used by ESIMD and Triton's SIMD path (VectorCompute intrinsics) | the driver's IGC; `ocloc` offline |
| 4a. Host: OpenCL / Level Zero / SYCL | the CM kernel is a SPIR-V module or binary, launched with SIMD1 dispatch (NDRange = threads) | the normal route; `examples/host_*.cpp` |
| 4b. Host: **CM runtime** (the "tasking layer") | `CmDevice` → `CmProgram` (`.isa`) → `CmKernel` → `CmThreadSpace` (2D thread grid, optional dependency patterns / wavefronts, `get_thread_origin_x/y`) → `CmTask` (a group of kernels) → `CmQueue::Enqueue` → `CmEvent`; surfaces are `CmBuffer`/`CmSurface2D` | media-driver runtime: Gen12 and older, **not DG2/Arc**; its `.isa` needs the 2021 cmc (`legacy-sdk.md`): `examples/cmrt/` |
| 5. Emulation | kernel compiled by g++ against `libcm`, run on the CPU through `libigfxcmrt_emu` (CM runtime API) or `libshim` (OpenCL C API) | 2021 SDK only (`legacy-sdk.md`): `examples/cmrt/`, `examples/emu/` |

## Where CM runs

**Intel GPUs only** (Gen9 and newer). "Metal" means close to the metal, not
Apple's Metal API. There is no CPU code generation: the OpenCL CPU device
rejects CM SPIR-V ("unsupported capability 5617", VectorComputeINTEL). On a
CPU, CM runs only through the 2021 SDK's **emulator**: the kernel is
compiled as ordinary C++ against `libcm` and launched through a "shim"
OpenCL look-alike, a debugging aid (gdb/printf in kernels) from an archived
project (github.com/intel/cm-cpu-emulation). It needs the 2021 SDK
(`examples/emu/run_emu.sh`, `references/legacy-sdk.md`).

## Toolchain

| Item | What to use |
|---|---|
| CM compiler | `cmc` from `intel-igc-cm` (1.0.225: clang 11, `-march=<device>`), installed **with the driver release it belongs to** from Intel's GPU repository (`references/pipeline.md` §7). If the driver comes from elsewhere (a distribution or PPA without `intel-igc-cm`), `scripts/install_cm_toolchain.sh` installs cmc with its **matched IGC** in user space, plus wrappers |
| Backend / offline compiler | the driver's IGC VC backend (`-vc-codegen`) and `ocloc` of the **same release** as cmc for SPIR-V; device binaries also load on newer drivers. Check versions against https://dgpu-docs.intel.com (`pipeline.md` §7) |
| oneAPI | `icpx` for SYCL hosts and ESIMD; `llvm-spirv` in `compiler/latest/bin/compiler/` disassembles SPIR-V. oneAPI does **not** ship cmc |
| CM runtime (legacy) | `libigfxcmrt` from the media driver (`libigfxcmrt7 libigfxcmrt-dev intel-media-va-driver-non-free libva-dev`): Gen12 and older, not DG2; `.isa` and emulation need the 2021 SDK (`legacy-sdk.md`) |
| Upstream sources and spec | https://github.com/intel/cm-compiler, `clang/compute_sdk/docs/` (`cmlangspec/`, `cmcuserguide/`, `cmoclrt/`, `cmtutorial/`) |

## Quick start (tested)

```bash
S=<path to this skill>        # cmc and ocloc of one release on PATH (Toolchain above)
$S/scripts/cm_build.sh -d dg2 -o out $S/examples/vadd_genx.cpp   # cmc → .spv, ocloc → .bin (-d: your GPU)
source /opt/intel/oneapi/setvars.sh; icpx -fsycl -O2 -fp-model=precise -I$S/examples \
  $S/examples/host_sycl.cpp -lze_loader -lOpenCL -o host_sycl
./host_sycl out/vadd_genx.spv    # JIT from SPIR-V with -vc-codegen → PASS
./host_sycl out/vadd_genx.bin    # AOT binary (Level Zero interop) → PASS
$S/scripts/run_examples.sh -d dg2 cmtest   # build + check every example for one device
```

Work in a scratch directory; never write into an SDK or toolchain tree.

## Workflow for a new kernel

1. **Design per thread.** Decide what one hardware thread owns: a tile, a
   row block or a contiguous chunk of CHUNK elements. Size its register
   working set (vectors and matrices, total) to fit in the GRF. The default
   is 128 x 32 B = 4 KB on DG2; `-double-GRF` gives 8 KB at half the
   occupancy. Then pick the memory message that loads the tile in the fewest
   `send`s: 2D media block for images/tiles, OWord (or LSC) block for contiguous data, gathers.
   Write the sizes as `constexpr` constants (CHUNK, the message size BLOCK,
   tile sizes) with `static_assert`s tying them together. Move a chunk with an
   unrolled loop of maximum-size messages into `select` regions of one
   register vector, then compute on the whole chunk in one statement (see the
   skeleton below and `examples/vadd_genx.cpp`). The loop offsets become
   constants, so each piece is one `send` straight into its registers.
2. **Write the kernel** as `extern "C" _GENX_MAIN_ void name(...)` with every
   surface argument annotated (`[[type("buffer_t")]]`, `"svmptr_t"`,
   `"image2d_t"`). Compute thread ids from `cm_group_id`/`cm_local_id`/
   `cm_local_size`. Use `select`/`replicate`/`format`/`merge` to rearrange
   data in registers instead of round-tripping through memory.
3. **Compile** with `cmc -march=<device>` (or `scripts/cm_build.sh`, `-k` to keep IR/vISA/asm
   dumps). SPIR-V (`-emit-spirv`) is finalized by the driver when the program loads and runs on
   any Intel GPU with the VC backend; a device binary (`.bin`, or a fat binary for several GPUs,
   `targets.md` §6) skips that step but is tied to its targets. State the input sizes the kernel
   can't take (n not a multiple of CHUNK, not a power of two, ...) and how to lift them: padding,
   masking the last thread, or a tail kernel (`memory.md` §7).
4. **Host:** any of the **OpenCL C API**, the **OpenCL C++ bindings** (CLHPP,
   `examples/cm_ocl.hpp`) or **SYCL** (`examples/cm_sycl.hpp`: loads .spv/.bin/.cpp, USM
   arguments, event timing) runs CM kernels. Use what the request or the existing code uses, and
   ask when nothing decides it. SYCL can't pass `image2d_t` surfaces or set spec constants by id:
   use OpenCL for those. Dispatch threads with the SIMD1 model: global size = number of CM
   threads, local size = threads per group.
   The `cmsycl::`/`cmocl::` helpers belong to this skill, not to CM: a CM kernel is just a SPIR-V module,
   so any Intel GPU whose driver accepts SPIR-V and has the IGC VC backend runs it (`pick_gpu`).
   CM's own host API, the CM runtime (`libigfxcmrt`), supports Gen12 and older but not DG2:
   keep it for legacy code (`examples/cmrt/`).
5. **Check correctness against a CPU reference.** Pick test data whose
   reference result is exact (small integers, sums of representable values),
   so mismatches are real bugs and not rounding noise.
6. **Inspect performance:** `cm_build.sh -k`, then read
   `igc_dump/*_<kernel>.asm` for `numGRF`, spills, number of `send`s and
   unfolded `mov`s (`references/pipeline.md` section 6).

## Kernel skeleton

```cpp
#include <cm/cm.h>              // implicit with cmc, harmless to include

constexpr int CHUNK = 64;       // floats per thread
constexpr int BLOCK = 32;       // floats per OWord block message (128 B maximum)
static_assert(CHUNK % BLOCK == 0, "a chunk must be whole messages");

extern "C" _GENX_MAIN_ void saxpy(SurfaceIndex x [[type("buffer_t")]],
                                  SurfaceIndex y [[type("buffer_t")]],
                                  float a) {
  uint tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);  // thread index
  uint off = tid * CHUNK * sizeof(float);    // byte offset, 16 B aligned
  // n must be a multiple of CHUNK: the host pads x and y (Gotchas, "array tails")
  vector<float, CHUNK> vx, vy;
#pragma unroll
  for (int b = 0; b < CHUNK / BLOCK; ++b) {  // one send per 128 B piece
    read(x, off + b * BLOCK * sizeof(float), vx.select<BLOCK, 1>(b * BLOCK));
    read(y, off + b * BLOCK * sizeof(float), vy.select<BLOCK, 1>(b * BLOCK));
  }
  vy += a * vx;                              // one SIMD-CHUNK statement
#pragma unroll
  for (int b = 0; b < CHUNK / BLOCK; ++b) {
    write(y, off + b * BLOCK * sizeof(float), vy.select<BLOCK, 1>(b * BLOCK));
  }
}
```

## Host launch

**SYCL** (`examples/cm_sycl.hpp`; every `host_*.cpp` without `_ocl`; tested with `.spv`, `.bin`
and `-cmc` source on the Level Zero and OpenCL backends):
```cpp
#include "cm_sycl.hpp"   // icpx -fsycl -O2 -fp-model=precise host.cpp -lze_loader -lOpenCL
sycl::queue q = cmsycl::make_queue();                         // Intel GPU, IN-ORDER, profiling
cmsycl::program prog = cmsycl::load_program(q, "saxpy_genx.spv");
//   .spv → kernel_compiler (spirv) + "-vc-codegen" [+ -ftranslate-legacy-memory-intrinsics on LSC GPUs]
//   .bin → Level Zero / OpenCL interop;  .cpp → "-cmc" (intel-igc-cm + matched IGC)
sycl::kernel saxpy = prog.kernel("saxpy");                    // extern "C" name
float *x = cmsycl::to_device(q, hx, pad), *y = ...;           // buffer_t / svmptr_t ← USM pointers
sycl::event e = cmsycl::launch(q, saxpy, n / CHUNK, 1, x, y, a);  // global size counts THREADS
double ms = cmsycl::elapsed_ms(e);                            // profiling event (waits first)
cmsycl::to_host(q, y, hy);                                    // launch2d(q, k, tx, ty, gx, gy, ...): CM order
```
- **USM has no bounds checks.** A `cl_mem` surface returns 0 for out-of-range reads and drops
  out-of-range writes, which CM kernels often rely on for a partial last block or thread. Pad USM
  allocations to whole blocks/threads (`host_spmv.cpp`, `host_gesv.cpp` lanes mode).
- **Order:** SYCL queues are out-of-order by default and USM carries no dependencies, so
  `make_queue` builds an in-order queue (a reordered memset/memcpy broke SpMV during the port).
- **`-fp-model=precise`:** icpx defaults to fast math, which changed `host_gesv`'s CPU reference.
- Don't gate on `ext_oneapi_can_compile(source_language::spirv)`: it wrongly returns false.

**OpenCL C++ bindings** (`examples/cm_ocl.hpp`), also needed for what SYCL can't pass: `image2d_t`
media-block surfaces (`cl::Image2D`, `host_linear_ocl.cpp`) and SPIR-V specialization constants by
id (`setSpecializationConstant`, `host_specconst_ocl.cpp`); `host_ocl.cpp` is the plain reference.
Same flow: `cmocl::pick_gpu()`, `cmocl::load_program(ctx, dev, "k.spv")`, then
`cl::KernelFunctor<cl::Buffer, cl::Buffer, float> k(prog, "saxpy"); k(cl::EnqueueArgs(q,
cl::NDRange(n / CHUNK), cl::NDRange(1)), dx, dy, a);`.

**OpenCL C API:** the same calls without the wrappers: `clCreateProgramWithIL` + `clBuildProgram(...,
"-vc-codegen")` (or `clCreateProgramWithBinary` for the `.bin`), `clCreateKernel`, `clSetKernelArg`
with the `cl_mem` (`clSetKernelArgSVMPointer` for `svmptr_t`), `clEnqueueNDRangeKernel` with the
global size in threads. Check every return code and print the build log on failure.

**Level Zero:** `zeModuleCreate` with `ZE_MODULE_FORMAT_IL_SPIRV` and `pBuildFlags = "-vc-codegen"`
(or `NATIVE` with the `.bin`); see `references/pipeline.md` §5.

**From CM source at run time:** pass the `*_genx.cpp` text itself with `-cmc [-D...]` (OpenCL
`clCreateProgramWithSource`, SYCL `source_language::opencl`, `ocloc -options "-cmc"`). Intel GPU
drivers only; needs the `intel-igc-cm` front end + matched IGC (`pipeline.md` §5).

**CM runtime (legacy tasking layer:** `CmDevice` … `CmQueue`, `libigfxcmrt`): Gen12 and older,
not DG2, and its `.isa` comes from the 2021 SDK. The full flow, adapter selection, print buffer
and emulation build are in `references/legacy-sdk.md` §6 (`examples/cmrt/`).

## Compilation pipeline

```
CM .cpp ──cmc -E──────────────▶ .i
        ──cmc -emit-llvm -S───▶ .ll   (front-end IR: llvm.genx.* intrinsics, unoptimized)
        ──cmc -emit-spirv─────▶ .spv  (SPV_INTEL_vector_compute; device independent)
.spv ──ocloc -spirv_input -options "-vc-codegen" -device dg2──▶ .bin (zebin)
     └─ or JIT: clCreateProgramWithIL / zeModuleCreate / SYCL kernel_compiler + "-vc-codegen"
     inside IGC: GenX LLVM passes (rdregion/wrregion, baling, legalization) → vISA → finalizer → Gen asm
IGC_ShaderDumpEnable=1 IGC_DumpToCurrentDir=1 <any of the above>
     → VC_*_optimized.ll, *.visaasm, *.asm, *.zeinfo
```
Commands for every stage:
```bash
cmc -march=dg2 -emit-spirv -o k.spv k_genx.cpp     # -march: dg2, rkl, mtl-h, bmg, ... (targets.md)
cmc -march=dg2 -emit-llvm -S -o k.ll k_genx.cpp
cmc -march=dg2 -o k.bin k_genx.cpp                 # straight to a zebin
/opt/intel/oneapi/compiler/latest/bin/compiler/llvm-spirv -to-text k.spv -o k.spvasm
ocloc compile -file k.spv -spirv_input -device dg2 \
      -options "-vc-codegen -ftranslate-legacy-memory-intrinsics" -output k -output_no_suffix
# use the ocloc of cmc's own release: pipeline.md §7, "pairing rule"
```
Option tables, dump files, upstream `-march` and inspection tips are in
**`references/pipeline.md`**. To read the generated vISA and Gen assembly
(regions, SWSB tokens, LSC vs legacy sends, spills), and for the tested
IGC dump flags (`IGC_DumpToCustomDir`, `IGC_ShaderDumpRegexFilter`, ...),
see **`references/assembly.md`**.

## Language essentials

Full reference, including built-ins, SIMD control-flow rules, functions and
globals: **`references/language.md`**.

```cpp
vector<float, 16> v;  matrix<uchar, 8, 32> m;              // register objects
vector_ref<float, 8> odd = v.select<8, 2>(1);              // view: 8 elems, stride 2, from 1 (l-value)
auto win = m.select<6, 1, 24, 1>(1, 3);                    // 6x24 sub-matrix at (1,3)
m.row(2) = 0;  t.row(c) = m.column(c);                     // row/column views
vector_ref<int, 64> raw = m.format<int>();                 // reinterpret bits (no copy)
vector<int, 8> g = v8.iselect(idx);                        // register gather (r-value)
vector<int, 32> r = v.replicate<8, 4, 4, 0>(0);            // REP,VS,W,HS regions
v.merge(x, v > 0);  v.merge(x, y, mask);                   // predicated mov / select
if ((m == 0).any()) { ... }                                // mask → scalar branch
vector<uchar, 16> s = vector<uchar, 16>(vi, SAT);          // saturating convert
float t = cm_sum<float>(v);                                // reductions, cm_max/min/abs/...
SIMD_IF_BEGIN(v < 0) { v = -v; } SIMD_ELSE { v *= 2; } SIMD_IF_END;   // per-lane branches
```
Rules to keep in mind:
- Template sizes and strides are compile-time. Offsets may be variables but
  then compile to indirect addressing, so `#pragma unroll` loops whose
  counter feeds a select offset. `#pragma unroll` does **not** make the
  counter a constant expression: a size such as `select<N - s, 1>` with
  loop variable `s` fails to compile. Use `if constexpr` template recursion
  (`examples/esimd/pipeline_functor.cpp`, `scan_step<S>`).
- No pointers, arrays, references or recursion in device code (structs,
  templates, `constexpr` and `static_assert` are fine). Pass by
  `vector_ref`/`matrix_ref` for out-parameters.
- Mixed shapes are fine if element counts match (row-major); operands are read before writing.
- SIMD control-flow blocks need a width that is a power of two up to 32,
  allow no function calls, no block or media I/O and no scalar `if` inside.
  For short bodies `merge` is usually cheaper.
- Run-time constants: `cm_spec_constant<T, id, default>()` or `-D`/templates/
  scalar arguments: **`references/language.md` §8, "Specialization constants"**.
- Per-GPU code: `CM_HAS_LSC`/`CM_HAS_DPAS`, `__CM_INTEL_TARGET_ID >= __CM_INTEL_TARGET(12,55,0)`,
  legacy `CM_GENX`; codes, tables and pitfalls: **`references/targets.md`**.

## Memory essentials

Full table of messages, alignments, limits and out-of-bounds behaviour:
**`references/memory.md`**.

| Need | Use (legacy dataport API: every cmc and GPU) |
|---|---|
| contiguous ≤128 B | `read(buf, byte_off, v)` / `write(...)`: OWord block, 16 B aligned |
| 2D tile of an image | `read(img, x_bytes, y_rows, matrix)`: media block, up to 64 B x 4 ... 4 B x 64; edge-clamped reads; annotate `image2d_t` |
| gather/scatter | `read(buf, base, vector<uint,N> elem_offs, v)`, `read_scaled` for byte offsets |
| atomics | `write_atomic<ATOMIC_ADD, uint>(buf, dword_offs, src)`: use SIMD16 messages |
| raw pointers | `svmptr_t` + `cm_svm_block_read/write`, `cm_svm_scatter_*` |
| group sharing | `cm_slm_init`, `cm_slm_alloc`, `cm_slm_block_read/write`, `cm_slm_fence` + `cm_barrier` |
| XMX | `cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, M>(acc, b_vnni, a)`, N = 8 on DG2 |
| LSC (`cm_load<T,64>`, `cm_ptr_load`, `cm_atomic`; byte offsets) | DG2+ (`examples/lsc_genx.cpp`, `memory.md` §6); 2D LSC needs PVC/Xe2 |

## Gotchas (tested on DG2 and Gen12LP)

- **Legacy atomics: use SIMD16 messages.** A SIMD32 `write_atomic` hangs DG2 (or `CL_OUT_OF_RESOURCES`)
  unless the backend gets `-ftranslate-legacy-memory-intrinsics`, and loses half the increments on
  Gen12LP. SIMD16 works on both without the flag. The flag is a safe default on DG2+ but breaks
  pre-LSC builds; `cm_build.sh` (by `-d`) and `cm_ocl.hpp` (by IP version) add it only for DG2+.
- **Media-block images:** annotate them `[[type("image2d_t")]]`, the spelling every cmc reads as a
  media-block image.
- **OWord block limit:** 128 bytes per `read`/`write`; larger chunks use the skeleton's unrolled BLOCK loop.
- **CLHPP coarse-grained SVM** (`cl::SVMAllocator<T, cl::SVMTraitCoarse<>>`) comes **already mapped**:
  fill it, `cl::unmapSVM(v)` before the launch, `cl::mapSVM(v)` to read (after `setDefault(q)`).
- **Units differ:** OWord block, `read_scaled`, media x, every LSC and ESIMD offset: bytes;
  scattered `read`/`write`: elements; `write_atomic`: DWords; media y: rows.
- **Array tails: pad the arrays.** Block messages move whole blocks, so when n isn't a multiple of
  a thread's CHUNK the last thread reads and **writes past element n**. A `cl_mem` drops only writes past the end of the allocation; USM checks nothing, so the
  overrun corrupts the next allocation. Allocate `round_up(n, CHUNK)` (or whole groups), zero the
  pad, or store the tail with scattered writes (`memory.md` §7, "Array tails").
- **Dispatch counts threads.** A global size of n elements instead of n/32 launches 32x too many
  threads that access out of bounds (dropped/zero when stateful, undefined for SVM). The SDK's own
  `BufferTest` does this and passes only thanks to the bounds checks.
- `[[type("svmptr_t")]]` may draw an "ignored" warning; harmless, `clSetKernelArgSVMPointer` works.
- **Inter-thread communication:** atomics for flags and values (L1 isn't coherent), `cm_fence` between
  value and flag (costly on DG2), atomic tickets, and a bound on every spin: an endless spin **hangs
  the GPU** (context reset). See `scan_lookback`.
- **SLM handles:** `cm_slm_alloc` returns a handle. Pass the slot offset as the offset argument
  (`cm_slm_block_write(slm, lid * 16, v)`). `slm + lid * 16` is silently ignored by older cmc
  (every thread hits slot 0), so never add offsets to the handle. Call `cm_slm_init`/`cm_slm_alloc` in the kernel body only (`memory.md` §4).
- **`constexpr` offsets:** bind `constexpr`-function results to `constexpr` variables before using
  them as offsets; used directly they become a run-time call plus indirect addressing.
- **`cm_fbh(x)` on unsigned input returns the count of leading zeros**, not
  the bit index: the highest set bit is `31 - cm_fbh(x)`.
- **Signed zeros in max/min:** hardware `max` ranks -0.0 below +0.0 (also when the compiler turns a
  compare + `merge` into `max`). In an argmax, re-read the element at the chosen index.
- Global work offset is not supported. Stateful surfaces are limited to 4 GB.
- **cmc needs the IGC of its own release.** `intel-igc-cm` 1.0.225's SPIR-V fails on a newer
  IGC such as 2.42 ("verify Module failed"); its `.bin` files load fine. Install compiler and driver from the
  same release (`pipeline.md` §7).

## ESIMD (SYCL Explicit SIMD)

For new code inside a SYCL application, ESIMD gives the same model (`simd<T,N>`, `simd_view`,
`block_load/store`, `gather`, `atomic_update`, `slm_*`, `xmx::dpas`, `invoke_simd`) with no separate
toolchain. **`references/esimd.md`** has the CM→ESIMD API mapping, build flags (including
`invoke_simd`'s IGC env vars), the upstream examples validated on DG2 (`examples/esimd/`),
kernel forms (lambda vs functor, fused `ESIMD_INLINE` stages) and where to learn it (spec,
`sycl/test-e2e/ESIMD`, the oneAPI GPU Optimization Guide's explicit-SIMD chapter, which applies to CM too).

## Learning resources

The spec (github.com/intel/cm-compiler, `clang/compute_sdk/docs/cmlangspec/`), the CGO 2021 paper
(arXiv:2101.11049) and talk, and for new code the SYCL ESIMD docs: **`references/documentation.md`**.

## Bundled files

The examples were tested on DG2 (Arc A770) and Gen12LP (UHD 750); **`references/examples.md`**
lists each with what it shows and its measured results. Starting points:
- **Minimal kernel and hosts:** `vadd_genx.cpp` with `host_sycl.cpp` (SYCL), `host_ocl.cpp` (CLHPP).
- **Helpers:** `cm_sycl.hpp` and `cm_ocl.hpp`.
- **Idioms:** `patterns_genx.cpp` (2D threads, atomics, SIMD control flow, SLM, SVM).
- **Paper workloads:** `linear_genx.cpp` (media blocks), `spmv_genx.cpp`, `scan_genx.cpp`.
- **Dense and batched linear algebra:** `gesv_genx.cpp` / `gesvb_genx.cpp`, `dpas_genx.cpp` (XMX).
- **Reductions and APIs:** `argmax_genx.cpp`, `lsc_genx.cpp`, `target_genx.cpp`, `specconst_genx.cpp`, `traits_genx.cpp`.
- **Legacy:** `cmrt/` (CM runtime) and `emu/`, both with the 2021 SDK; `esimd/` (SYCL ESIMD).
- **Scripts:** `cm_build.sh` (cmc → SPIR-V → ocloc, `-k` dumps), `run_examples.sh` (everything), `install_cm_toolchain.sh`.
- **References:** `language.md`, `memory.md`, `pipeline.md`, `targets.md`, `assembly.md`, `esimd.md`, `simt-vs-simd.md`, `documentation.md`, `legacy-sdk.md`, `examples.md`.

Start new kernels from the closest example, since those are tested. When you change an example or write a new one, add a CPU-reference
check in the same style and run it on the GPU before reporting success.
