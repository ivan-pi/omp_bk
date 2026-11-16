# From C for Metal to SYCL ESIMD

`sycl_ext_intel_esimd` (Explicit SIMD) is CM's model inside DPC++/SYCL.
A kernel lambda marked `[[intel::sycl_explicit_simd]]` (or `SYCL_ESIMD_KERNEL`)
is one hardware thread operating on `simd<T, N>` register values. The
same IGC VC backend compiles both. The paper notes that DPC++-ESIMD
"integrates some CM language features into DPC++ ... and relies on CM
optimizations and code generation."

Headers (default oneAPI install):
`/opt/intel/oneapi/compiler/<version>/include/sycl/ext/intel/esimd/`.

**Where to learn ESIMD** (these are the links the Intel C++ developer
reference page "Explicit SIMD SYCL Extension" points to; Intel-hosted URLs
change with each release, so search for the title if one is dead):
- Specification: https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd
  (`sycl_ext_intel_esimd.md` and the API docs next to it).
- Examples, as Markdown with the build line and one C++ block each:
  https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/examples/README.md
  The six of them were validated for this skill; see section 6.
- **End-to-end tests (the largest example set):**
  https://github.com/intel/llvm/tree/sycl/sycl/test-e2e/ESIMD has 122 files
  plus subdirectories (`api/`, `dpas/`, `lsc/`, `kmeans/`, `linear/`,
  `mandelbrot/`, `named_barriers/`, `InlineAsm/`, `PerformanceTests/`, ...).
  Many are ESIMD ports of the classic CM tutorial workloads, so they are the
  best guide for translating a CM kernel: `BitonicSortK*.cpp`,
  `PrefixSum.cpp`, `Prefix_Local_sum{1,2,3}.cpp`, `histogram*.cpp` (including
  SLM and 2D variants), `matrix_transpose*.cpp`, `radix_sort.cpp`,
  `reduction.cpp`, `Stencil.cpp`/`stencil2.cpp`, `linear/` (the CM paper's
  filter).
- **oneAPI GPU Optimization Guide**, chapter "Optimizing Explicit SIMD
  Kernels" (2025.2):
  https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-explicit-simd-kernels.html
  (guide overview: .../optimization-guide-gpu/2025-2/overview.html). Its
  advice applies to CM almost word for word; the highlights are in section 5.

## Contents
1. Concept and API mapping table
2. Differences that bite
3. Build and run (JIT, AOT, invoke_simd)
4. Kernel forms and pipelines
5. Optimization guide highlights (ESIMD and CM)
6. Validated examples (`examples/esimd/`)

---

## 1. Mapping

| C for Metal | ESIMD (`namespace sycl::ext::intel::esimd`) |
|---|---|
| `_GENX_MAIN_ void k(...)` kernel, one thread | `q.parallel_for(range, [=](id<1> i) SYCL_ESIMD_KERNEL {...})`, one work-item = one thread |
| `_GENX_ inline` helper | `ESIMD_INLINE` / `SYCL_ESIMD_FUNCTION` functions |
| `cm_group_id`, `cm_local_id`, `cm_linear_global_id()` | `nd_item` / `id` of the lambda (`item.get_group(0)`, `item.get_local_id(0)`, `i[0]`) |
| `vector<T, N>` | `simd<T, N>` |
| `matrix<T, R, C>` | `simd<T, R*C>` plus `.bit_cast_view<T, R, C>()` (2D view) |
| `vector_ref`, `matrix_ref` | `simd_view` (returned by `select`, `bit_cast_view`, `row`, `column`) |
| `v.select<S, ST>(i)` | `v.select<S, ST>(i)` |
| `m.select<VS, VST, HS, HST>(i, j)` | `v.bit_cast_view<T, R, C>().select<VS, VST, HS, HST>(i, j)` |
| `m.row(i)`, `m.column(j)` | `view.row(i)`, `view.column(j)` |
| `v.format<U>()`, `m.format<U, R, C>()` | `v.bit_cast_view<U>()`, `v.bit_cast_view<U, R, C>()` |
| `v.iselect(idx)` | `v.iselect(simd<uint16_t, K> idx)` |
| `v.replicate<REP, VS, W, HS>(i)` | `v.replicate_vs_w_hs<REP, VS, W, HS>(i)` (also `replicate_w`, `replicate`) |
| `v.merge(x, mask)`, `v.merge(x, y, mask)` | `v.merge(x, mask)`, `v.merge(x, y, mask)` / free `merge(a, b, mask)` |
| `vector<ushort, N>` mask, `.any()`, `.all()` | `simd_mask<N>`, `.any()`, `.all()` |
| `vector<uchar,N>(v, SAT)` | `saturate<uchar>(v)` / `convert<...>` |
| `cm_sum<T>(v)`, `cm_reduced_max<T>(v)` | `reduce<T>(v, std::plus<>())`, `hmax<T>(v)`, `hmin<T>(v)` |
| `cm_bfn<expr>(a, b, c)` | `bfn<expr>(a, b, c)` |
| `cm_dpas<P1, P2, 8, M>(C, B, A)` | `xmx::dpas<8, M, ResT, CT, BT, AT, BPrec, APrec>(C, B, A)` (`sycl/ext/intel/esimd/xmx/dpas.hpp`) |
| `SIMD_IF_BEGIN ... SIMD_IF_END` | none: ESIMD has no SIMD control flow macros. Use `merge`/`simd_mask`, or ordinary uniform `if` with `.any()` |
| OWord `read(buf, off, v)` / `write` | `block_load<T, N>(usm_ptr + i)` / `block_store(ptr, v)`; accessor overloads take a byte offset |
| scattered `read(buf, g, offs, v)` (element offsets) | `gather<T, N>(ptr, simd<uint32_t, N> byte_offsets)`: **byte** offsets |
| `write_atomic<ATOMIC_INC, uint>(buf, dw_offs)` | `atomic_update<atomic_op::inc, uint32_t, N>(ptr, byte_offsets [, mask])` |
| media block `read(img, x, y, m)` | `media_block_load<T, R, C>(image_acc, x_bytes, y)` / `media_block_store` |
| `svmptr_t` + `cm_svm_block_read` | plain USM pointers + `block_load` |
| `cm_slm_init(n)`, `cm_slm_alloc` | `slm_init<n>()` (compile-time size) or a `local_accessor` |
| `cm_slm_block_read/write` | `slm_block_load<T, N>(byte_off)` / `slm_block_store` |
| `cm_slm_read`/`write` (gather/scatter) | `slm_gather` / `slm_scatter`, `slm_atomic_update` |
| `cm_slm_fence(...)`, `cm_barrier()` | `fence<...>()`, `barrier()` |
| LSC `cm_load<T,N,DS,L1,L2>` / 2D block | `block_load` with properties (`cache_hint_L1<...>`, `alignment<...>`), `load_2d` / `prefetch_2d` (`experimental` LSC) |
| `-ftranslate-legacy-memory-intrinsics` | not needed: ESIMD picks LSC or legacy messages per target |
| `_GENX_FLOAT_CONTROL_` | no equivalent (rounding mode set by `fp_model` compiler flags) |
| `printf` | no ESIMD-specific printf in the headers; debug by copying values to a USM buffer |

## 2. Differences that bite

- **Offsets are in bytes** for ESIMD `gather`/`scatter`/`atomic_update`. CM
  scattered read/write use element units and CM atomics use DWords.
- `using namespace sycl::ext::intel::esimd;` brings in `esimd::info`, which
  collides with `sycl::info`. Write `sycl::info::device::name` in full. Two
  upstream examples fail to compile because of this.
- **XMX ExecSize is target-specific.** It is 8 on DG2/Arc A-series and 16 on
  PVC and Xe2. The upstream `dpas` example is written for PVC: on DG2 (tested on an
  Arc A770) it compiles and runs with no error, but the results are wrong until
  `ExecSize = 8`. Choose the value per device (for example with
  `sycl::ext::oneapi::experimental::architecture` or a template on N), and
  check the results.
- `slm_init<N>()` needs a compile-time size and must come first in the
  kernel.
- ESIMD kernels and SPMD kernels can live in one program. Calling ESIMD from
  SPMD code goes through `invoke_simd` (experimental, see below).

## 3. Build and run

```bash
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -O2 sum_two_arrays.cpp -o sum_two_arrays                     # JIT (SPIR-V inside)
icpx -fsycl -O2 -fsycl-targets=spir64_gen -Xs "-device dg2" x.cpp -o x   # AOT for DG2
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./sum_two_arrays                   # or opencl:gpu
```
A compile of one of these examples peaked at about 225 MB of memory.

**invoke_simd** (calling an ESIMD function from SPMD code):
```bash
icpx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr invoke_simd.cpp -o invoke_simd
IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 ./invoke_simd   # JIT: env needed at run time
# AOT: set the same two variables at COMPILE time (ocloc reads them, else "error code -11");
# the AOT binary then runs without them.
```
Without those variables the JIT build fails with "VLD: Failed to compile
SPIR-V ... ZeBinary did not contain any .visaasm sections!".

## 4. Kernel forms and pipelines

- **Inline lambda** (`q.parallel_for(range, [=](...) SYCL_ESIMD_KERNEL {...})`):
  shortest, and best for a one-off kernel such as a port of a single CM
  kernel. Everything sits next to the launch.
- **Functor struct** (`struct K { ptrs...; void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {...} };`
  then `q.parallel_for(nd_range, K{...})`): better when a kernel is reused,
  templated on sizes or types, or assembled from parts.
- **Fused stages inside one kernel:** write each step as an `ESIMD_INLINE`
  function from `simd` to `simd`, then load once, run the stages on
  registers and store once. This is the CM way of building small
  factorizations or operation pipelines per hardware thread, and it saves a
  memory round trip per stage.
- **Kernels in sequence:** launch several ESIMD (or SPMD) kernels on an
  in-order queue, or chain them with events. Each boundary goes through
  memory, so use this when stages need a different decomposition or a
  global synchronization.
- `examples/esimd/pipeline_functor.cpp` shows both: a functor kernel fusing
  affine, clamp (`merge`) and an in-register prefix scan, then a lambda
  kernel that gathers each chunk's total. It passed on an Arc A770 (Level Zero
  and OpenCL).
- Sizes and strides of `select<>`/`replicate` are template arguments. Loop
  over them with `if constexpr` template recursion; `#pragma unroll` alone
  leaves the counter a run-time value, and the code does not compile.

## 5. Optimization guide highlights (ESIMD and CM)

From "Optimizing Explicit SIMD Kernels" (oneAPI GPU Optimization Guide
2025.2), with the CM equivalent where it differs:
- **Registers:** spills are the big slowdown. Size `simd`/`vector` objects
  per code region, update sub-vectors in place with `select` (no
  temporaries), and reshape with `bit_cast_view`/`format` and
  `replicate_vs_w_hs`/`replicate` to save instructions. For bank-conflict
  tuning, pin globals to GRF byte offsets: `ESIMD_REGISTER(n)` (CM:
  `_GENX_VOLATILE_BINDING_(n)`).
- **SIMD width:** use sizes that split into legal widths (48 = 32 + 16;
  47 splits into 32+8+4+2+1), round odd sizes up, and avoid very small
  vectors (SIMD8 reaches only half of peak on Max Series).
- **Cross-lane sharing:** load a block once and reuse its neighbours through
  `select`, as in the 3x3 linear filter (an 8x32 block in, 6x24 out, rows
  padded for layout). Prefer `media_block_load`/2D block loads to
  gather/scatter.
- **Indirect register access** (a variable index into a register vector) is
  slower than direct access, but faster than gathers or SLM. `#pragma
  unroll` loops with constant bounds so indices become direct.
- **Conditions:** replace per-lane `if/else` with `simd_mask` and `merge`,
  and use `any()`/`all()` to skip rarely needed work.
- **Math:** native math intrinsics have hardware precision; `bfn<F>()`
  computes any 3-input boolean function in one instruction; `reduce<T>(v, op)`
  does not guarantee element order.
- **Memory:** block loads beat per-lane gathers (more data per message,
  smaller payloads, fewer address instructions). On LSC hardware use
  `config_2d_mem_access` with `lsc_load_2d`/`lsc_store_2d`, the 2D
  **transform (VNNI)** mode for FP16/INT8 `dpas` operands, `lsc_prefetch_2d`
  ahead of use, and cache hints as template parameters. `copy_from`/`copy_to`
  let the compiler pick the message, and `address_space_cast` avoids runtime
  generic-pointer checks.
- The page does not cover SLM, atomics, large-GRF flags or `invoke_simd`;
  for those see sections 3-4 of this file and the e2e tests above.

## 6. Validated examples (`examples/esimd/`)

Tested with icpx 2026.1.0 on an Arc A770 (DG2) under Level Zero and
OpenCL. Every example prints the device name.

| File | Shows | Result | Fix applied |
|---|---|---|---|
| `sum_two_arrays.cpp` | `simd<float,VL>` load/add/store with USM and an accessor | PASS (JIT and AOT) | `info::` → `sycl::info::` |
| `simd_view.cpp` | `select` views of views, `bit_cast_view` | PASS | same |
| `bfn.cpp` | `bfn<F>(x, y, z)` 3-input boolean | PASS | none |
| `dpas.cpp` | `xmx::dpas` u4 x u4 → u32 | PASS (JIT and AOT) | `ExecSize = 8` (published value is 16, for PVC) |
| `invoke_simd.cpp` | SPMD → ESIMD call with `uniform` | PASS | needs the flags and env above |
| `invoke_simd_slm.cpp` | `local_accessor` passed into ESIMD `block_load/store` | PASS | same; prints harmless "Undefined function `__builtin_invoke_simd`" warnings |
| `pipeline_functor.cpp` (this skill) | functor-struct kernel with fused ESIMD_INLINE stages, then a lambda kernel | PASS (L0 and OpenCL) | — |

The upstream examples exist as Markdown (`examples/*.md`) with the C++ in
one code block; the `.cpp` files in `examples/esimd/` are those blocks plus the fixes above.
