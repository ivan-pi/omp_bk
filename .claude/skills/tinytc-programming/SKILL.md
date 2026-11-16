---
name: "tinytc-programming"
description: "Write, compile and run Intel Tiny Tensor Compiler (tinytc) kernels: tensor language, tinytc/tinytc-opt CLI, C/C++ core and SYCL APIs. Use for any tinytc build, kernel or host-integration task."
---

# Tiny Tensor Compiler (tinytc) programming

tinytc (https://github.com/intel/tiny-tensor-compiler, docs: https://intel.github.io/tiny-tensor-compiler/)
compiles programs in an MLIR-like **tensor language** to **SPIR-V** and launches them on Intel GPUs via
**SYCL**, Level Zero, or OpenCL. Its strength is small/odd-shaped GEMMs, batched tensor contractions and
fused kernels built from BLAS-like collective instructions (`gemm`, `ger`, `axpby`, …) plus explicit
subgroup-level code with cooperative matrices (XMX).

This skill was written against **v0.5.0** (Nov 2025; master at commit e29f6f4, Jan 2026). The language
and API changed a lot between 0.3 → 0.4 (OpenCL-C → SPIR-V, immediate operands removed, instructions
annotate *return* types). If the user is on another version, check `CHANGELOG.md` before trusting syntax here.

## Workflow

1. Write the kernel as a `.ir` text file (or a string in host code).
2. Validate it offline with `tinytc-opt` / `tinytc` (no GPU needed — see "CPU-only build" below).
3. In the host program: parse → JIT to a SYCL `kernel_bundle` → `create_kernel` → `get_execution_range`
   → `h.set_args(...)` following the **calling convention** → `h.parallel_for`.
4. Compile once, launch many times — JIT is expensive relative to a launch.

When writing tinytc code for the user, always run the kernel through `tinytc-opt -pcheck-ir` (and ideally
`tinytc -d <arch>`) before handing it over. The parser gives precise, located error messages.

---

## 1. Setup and building

### Dependencies
- Required: CMake ≥ 3.23, **re2c ≥ 3.0**, **bison ≥ 3.8.2**, a C++20 compiler.
- SYCL runtime: Intel oneAPI Base Toolkit (`icpx`), plus OpenCL ICD loader and Level Zero loader.
  `BUILD_SYCL=ON` forces `BUILD_OPENCL=ON` and `BUILD_LEVEL_ZERO=ON`.

### Full build with oneAPI (SYCL)
```bash
. /opt/intel/oneapi/setvars.sh
git clone https://github.com/intel/tiny-tensor-compiler.git tinytc
cmake -S tinytc -B tinytc/build -GNinja -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_INSTALL_PREFIX=$PWD/install -DBUILD_SHARED_LIBS=YES
cmake --build tinytc/build
cmake --install tinytc/build
```
CMake options: `BUILD_SYCL` (default ON), `BUILD_LEVEL_ZERO`, `BUILD_OPENCL`, `BUILD_TESTING`,
`BUILD_DOUBLE_PRECISION_TESTS` (turn off for iGPUs without fp64), `BUILD_DOCUMENTATION`,
`BUILD_SHARED_LIBS`.

### CPU-only build (for writing/validating kernels without a GPU)
The core library and both CLI tools build with any C++20 compiler, no oneAPI needed. Useful in a sandbox:
```bash
sudo apt-get update && sudo apt-get install -y re2c bison ninja-build   # Ubuntu 24.04: re2c 3.1
cmake -S tinytc -B tinytc/build -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SYCL=OFF -DBUILD_OPENCL=OFF -DBUILD_LEVEL_ZERO=OFF -DBUILD_TESTING=OFF
cmake --build tinytc/build -j
# binaries: build/tools/offline_compiler/tinytc, build/tools/opt/tinytc-opt
```
If apt has no re2c, `apt-get update` first; otherwise build re2c from github.com/skvadrik/re2c with CMake.

### Consuming tinytc from CMake
```cmake
find_package(tinytc REQUIRED)            # core: parser, compiler, builder, recipes
find_package(tinytc_sycl REQUIRED)       # or tinytc_ze / tinytc_cl; may add "shared" or "static"
target_link_libraries(app PRIVATE tinytc::tinytc tinytc::tinytc_sycl)
```
Pass `-DCMAKE_PREFIX_PATH=/path/to/install` for non-system installs. Compile the app with `icpx -fsycl`
(or use `find_package(IntelSYCL)` / `add_sycl_to_target`). A SYCL app needs only the second package:
`find_package(tinytc_sycl CONFIG)` pulls in the core one, and linking `tinytc::tinytc_sycl` alone
is enough (verified with v0.5).

A tinytc built with `icpx` links oneAPI runtime libraries (e.g. `libsvml`), so even the CLI tools
(`tinytc-opt`, `tinytc`) need the oneAPI environment (`setvars.sh`) and the tinytc `lib/` on
`LD_LIBRARY_PATH`; otherwise they fail to load.

### Headers
| Header | Contents |
|---|---|
| `tinytc/tinytc.hpp` | umbrella: core + builder (C++) |
| `tinytc/core.hpp` | compiler context, parse, core info, compile, binary, passes |
| `tinytc/builder.hpp` | IR builder, recipes (`create_small_gemm_batched`, …), `to_type<T>` |
| `tinytc/tinytc_sycl.hpp` | SYCL interop (C++ only) |
| `tinytc/tinytc_ze.h[pp]`, `tinytc/tinytc_cl.h[pp]` | Level Zero / OpenCL interop |
| `tinytc/tinytc.h` | C API |

---

## 2. Command-line drivers

All tools: positional `file-name` (omit → read stdin), `-o file` (omit → stdout).

### `tinytc` — offline compiler (.ir → SPIR-V)
```
tinytc [-d arch] [-O level] [-f flag]... [-F feature] [-S] [-o out] [file.ir]
```
- `-d` target: `tgl`, `pvc` (default), `bmg`, `bmg_g31`, `lnl`, `ptl_u`, `ptl_h`
  (`bmg_g31`, `lnl`, `ptl_*` are experimental). Unknown name → `Error (2): Invalid arguments`.
  **Arc A-series (DG2/Alchemist) has no target:** `acm-g10`, `dg2` and `xe_hpg` are all rejected. To
  check a DG2 kernel offline use `-d tgl` (same sub-groups 8/16/32, 32-byte registers, no XMX/2-D
  block IO tables); on the device, JIT through SYCL/Level Zero with `create_core_info(device)`, which
  works on an A770 at `support_level::basic` (only PVC and BMG report `tuned` in v0.5).
- `-O` optimization level (default 2).
- `-f unsafe-fp-math` / `-f no-unsafe-fp-math` — optimizer flags.
- `-F large-register-file` — core feature (on PVC: 2× GRF, ½ hardware threads; kernel compiled with
  `-ze-opt-large-register-file`). If you use it offline, pass the same flag at load time (see §5.3).
- `-S` emit SPIR-V **assembly** text instead of a binary — good for inspecting
  `OpExecutionMode ... LocalSize / SubgroupSize` and parameter lists.

```bash
tinytc -d pvc -o kernel.spv kernel.ir
tinytc -d bmg -S kernel.ir | grep -E 'OpEntryPoint|OpExecutionMode'
```

### `tinytc-opt` — run passes, print IR
```
tinytc-opt [-d arch] [-p pass]... [-O level] [-f flag] [-F feature] [-o out] [file.ir]
```
With no `-p` it just parses and pretty-prints (normalizes syntax, expands `$` preprocessor variables).
Passes (`tinytc-opt --help` lists them):
`check-ir`, `constant-propagation`, `dead-code-elimination`, `insert-lifetime-stop`, `set-stack-ptr`,
`insert-barrier`, `work-group-size`, `lower-linalg`, `lower-foreach`, `lower-coopmatrix`,
`dump-ir`, `dump-control-flow-graph`, `dump-def-use`, `dump-gcd`.

The JIT pipeline runs, in order: check-ir → (O≥1: constant-propagation, DCE) → insert-lifetime-stop →
set-stack-ptr → insert-barrier → **work-group-size** → **lower-linalg** → set-stack-ptr → lower-foreach →
(CP, DCE) → lower-coopmatrix → check-ir → SPIR-V. When running passes manually keep that order:
`lower-linalg` before `work-group-size` fails with *Internal compiler error (Subgroup size is missing)*.

Useful invocations:
```bash
tinytc-opt -pcheck-ir k.ir                                          # syntax + type check
tinytc-opt -pinsert-lifetime-stop -pinsert-barrier -pwork-group-size k.ir   # see chosen sg/wg size, barriers, SLM reuse
tinytc-opt -d bmg -pwork-group-size -plower-linalg k.ir            # see how gemm/axpby are tiled
tinytc-opt -pdump-gcd k.ir                                          # what the alignment/GCD analysis knows
tinytc-opt -pinsert-lifetime-stop -pset-stack-ptr k.ir | grep alloca   # SLM offset of every alloca
```
The last one annotates each `alloca` with `; stack_ptr: <byte offset>`; allocas whose lifetimes do not
overlap share offsets, so the peak (largest offset + size) is the kernel's SLM footprint. Allocas are
freed at their last use, so declaring each temporary just before the step that fills it maximizes reuse.

The allocator is stack-like (first fit), so the peak can exceed the largest set of live arrays: in
one kernel it placed a buffer above another that was still live, giving 86.7 KB where 64.0 KB was ever
live at once. To control the layout, make one `alloca` and carve views out of it yourself
(views keep the `local` space; `set-stack-ptr` then shows the single offset):
```
%slm = alloca : memref<f32x$N x2,local>
%lo  = subview %slm[0:$N,0] : memref<f32x$N,local>         ; two halves, reused step by step
%hi  = subview %slm[0:$N,1] : memref<f32x$N,local>
%U   = expand %hi[0->$B x$Q x$Q x$Q] : memref<f32x$B x$Q x$Q x$Q,local>   ; $N = $B*$Q*$Q*$Q
```
Reusing a half for a new array needs a barrier between the step that last reads it and the one that
writes it; between `foreach_tile` steps place it yourself (§8).

### Device query tools (built with the runtimes)
`tinytc-ze-info`, `tinytc-cl-info`: list drivers/devices with subgroup sizes, register space and the
SPIR-V features tinytc detects (bf16 conversion, float atomics, block IO …). Run these first on a new
machine to see whether a kernel's types/features are supported.

---

## 3. Tensor language essentials

LLVM/MLIR-flavoured SSA. `%local`, `@global`, `;` comments. Every value-producing instruction annotates its
**result type** after `:`. There are **no immediate operands** — make constants with `constant`.

### Execution model (the key concept)
- A `func` is a kernel; it is launched as a batch of **work-groups** indexed by a 3-D `group_id`.
  Typical pattern: one work-group per batch entry, `%b = group_id.x : index`.
- Each work-group = 2-D grid of subgroups. Work-group size is 2-D `[x, y]` in work-items; `x` must be a
  multiple of the subgroup size. `num_subgroups.x = wg[0]/sgs`, `num_subgroups.y = wg[1]`.
- **Collective region** (function body default): instructions like `gemm`, `axpby`, `alloca`, `foreach`
  are executed by the whole work-group, work distribution is implementation-defined, and **barriers are
  inserted automatically** between linalg instructions (`gemm`, `axpby`, …). **Not** between two
  `foreach`/`foreach_tile` loops: if one loop writes SLM that the next reads, put `barrier.local`
  between them yourself (checked with `-pinsert-barrier`: it adds none there, as the pass skips SPMD
  bodies and records no memory effects for the foreach instruction itself; v0.5).
- **SPMD region** (`parallel { }`, `foreach`, `foreach_tile` bodies): OpenCL-style per-work-item code;
  you insert `barrier.local` / `barrier.global` yourself. Collective instructions are illegal inside.
- Mixed instructions (`load`, `store`, arithmetic, `for`, `if`, `subview`, …) work in both.

### Types
- Scalars: `bool`, `i8 i16 i32 i64 index`, `bf16 f16 f32 f64`, `c32 c64`. `index` is 64-bit.
- `memref<f32x4x?>` — column-major, packed by default; `?` = runtime size.
  Explicit layout: `memref<f32x8x4,strided<1,?>>`. Address space: `,local` (only from `alloca`).
  Order-0 `memref<f32>` is a pointer to one scalar.
- `group<memref<f32x9x9>x?>` — pointer-to-pointer batch (`float**`), optional `, offset: ?`.
- `coopmatrix<f16x32x16,matrix_a>` (uses `matrix_a`, `matrix_b`, `matrix_acc`) — subgroup-distributed
  tile; only inside SPMD code; never a kernel argument.
- Mixed precision is allowed only along the "promotable" partial order (e.g. f16→f32, i8→f32; never i64→f64).

### Function and argument attributes
```
func @k(%A: memref<f32x?x?> {alignment=64, shape_gcd=[16], stride_gcd=[1,16]},
        %B: group<memref<f64x8x8>x?>)
        attributes {subgroup_size=16, work_group_size=[32,2]} { ... }
```
`alignment` (bytes, multiple of element size), `shape_gcd`/`stride_gcd` (divisibility promises that unlock
vectorized/block IO). Omit `subgroup_size`/`work_group_size` to let the compiler choose.

### Instruction cheat sheet (v0.5)
Collective (linalg, all take `alpha ... beta ...` as **SSA values**, update the last operand in place):
```
gemm[.atomic][.n|.t][.n|.t] %alpha, %A, %B, %beta, %C   ; C := α op(A) op(B) + β C   (order-2 memrefs)
gemv[.atomic][.n|.t]        %alpha, %A, %x, %beta, %y
ger[.atomic]                %alpha, %x, %y, %beta, %C   ; C := α x yᵀ + β C
hadamard[.atomic]           %alpha, %x, %y, %beta, %z   ; elementwise product
axpby[.atomic][.n|.t]       %alpha, %A, %beta, %B       ; B := α op(A) + β B (copy when β=0)
sum[.atomic][.n|.t]         %alpha, %A, %beta, %b       ; row sums (vector b) or full sum (order-0 b)
cumsum[.atomic]             %alpha, %A, <mode>, %beta, %B
%t = alloca {alignment=64} : memref<f32x56x9,local>      ; static shape, SLM, freed at end of region
foreach (%i,%j)=(%c0,%c0),(%m,%n) { ... }                 ; unordered SPMD loop over a box
foreach_tile (%i,%j)=(%c0,%c0),(%m,%n) as (%ti,%tj)<=(32,32) { ... }  ; tile offsets + remainder sizes
parallel { ... }                                         ; open an SPMD region
```
`.atomic` requires β to be a constant 0 or 1.

Mixed:
```
%c = constant 1.0 : f32          %z = constant 0.0 : coopmatrix<f32x32x32,matrix_acc>   %t = constant true : bool
%g = group_id.x : index          num_groups.{x,y,z} : index   num_subgroups.{x,y} : i32   subgroup_size : i32
%v = load %A[%i,%j] : f32        %m = load %G[%b] : memref<f32x9x9>   ; scalar load or memref from a group
store %v, %A[%i,%j]
%s = subview %A[%off:%len, %j] : memref<f32x?>    ; offset:SIZE (not end!); single index drops the mode
%n = size %A[1] : index
%e = expand %y[0->2 x 8] : memref<f32x2x8>        ; %y: memref<f32x16>; SSA sizes make that mode '?'
%f = fuse %F[0,1] : memref<f32x16x8>              ; %F: memref<f32x4x4x8>
add sub mul div rem max min shl shr and or xor : T
abs neg not conj re im : T        exp exp2 log log2 sin cos native_* : T
equal not_equal less_than less_than_equal greater_than greater_than_equal : bool
%x = cast %y : f32
%r = if %cond -> (f32) { ... yield (%a) } else { ... yield (%b) }
%acc = for %k=%lo,%hi,%step init(%c=%c0) -> (f32) { ...; yield (%c2) } attributes {unroll=4}
barrier.local   barrier.global   atomic_load/atomic_store/atomic_add/min/max (with .device/.work_group … and .relaxed/.acquire …)
```
SPMD-only: `subgroup_id.{x,y} : i32`, `subgroup_linear_id`, `subgroup_local_id`,
`subgroup_broadcast`, `subgroup_reduce_{add,max,min}`, `subgroup_{in,ex}clusive_scan_*`,
`cooperative_matrix_{load,store}[.t][.rows_checked|.cols_checked|.both_checked]`,
`cooperative_matrix_mul_add`, `_scale`, `_apply`, `_reduce_{add,max,min}.{row,column}`,
`_extract`, `_insert`, `_construct`, `_prefetch`, `_atomic_{load,store,add,min,max}`.
Rows of A, C in `cooperative_matrix_mul_add` must be multiples of the subgroup size.

### Preprocessor (parser-level templating)
```
$T = f32
$N = 64
$NB = !calc($N 16 /)                         ; integer RPN calculator: + - * / % ^ min max
$attrs = {subgroup_size=16, work_group_size=[32,2]}
func @k(%A: memref<$T x $N x?>) attributes $attrs { ... }
```
`$` variables can stand for constants, types, or array/dictionary/string attributes. Host code typically
prepends `$T = f64\n$N = 64\n` to a template string before `parse_string` (see the flash_attention example).
Note the spaces: write `memref<$T x $N x?>` so the lexer can split tokens. `$` definitions are lost when IR
is printed and are not available through the builder API.

### Example 1 — fused batched matrix chain (collective, verified with tinytc 0.5)
```
; Q(:,:,e) += K * P(:,:,e) * A_e
func @fused_kernel(%K: memref<f32x56x56>,
                   %P: memref<f32x56x9x?>,
                   %A: group<memref<f32x9x9>x?>,
                   %Q: memref<f32x56x9x?>) {
    %gid = group_id.x : index
    %p = subview %P[0:56,0:9,%gid] : memref<f32x56x9>
    %a = load %A[%gid]             : memref<f32x9x9>
    %q = subview %Q[0:56,0:9,%gid] : memref<f32x56x9>
    %tmp = alloca : memref<f32x56x9,local>
    %c0 = constant 0.0 : f32
    %c1 = constant 1.0 : f32
    gemm.n.n %c1, %K, %p, %c0, %tmp
    gemm.n.n %c1, %tmp, %a, %c1, %q        ; compiler inserts barrier.local before this
}
```

### Example 2 — explicit XMX tiles (SPMD, verified for pvc and tgl targets)
```
func @bgemm(%A: memref<f16x64x64x?>, %B: memref<f16x64x64x?>, %C: memref<f32x64x64x?>)
    attributes {subgroup_size=16, work_group_size=[32,2]} {
    %b  = group_id.x : index
    %a  = subview %A[0:64,0:64,%b] : memref<f16x64x64>
    %bb = subview %B[0:64,0:64,%b] : memref<f16x64x64>
    %c  = subview %C[0:64,0:64,%b] : memref<f32x64x64>
    parallel {
        %sx = subgroup_id.x : i32
        %sy = subgroup_id.y : i32
        %sxi = cast %sx : index
        %syi = cast %sy : index
        %c0 = constant 0 : index
        %c32 = constant 32 : index
        %c64 = constant 64 : index
        %i0 = mul %sxi, %c32 : index
        %j0 = mul %syi, %c32 : index
        %acc0 = cooperative_matrix_load %c[%i0,%j0] : coopmatrix<f32x32x32,matrix_acc>
        %acc = for %k=%c0,%c64,%c32 init(%cc=%acc0) -> (coopmatrix<f32x32x32,matrix_acc>) {
            %ta = cooperative_matrix_load %a[%i0,%k]  : coopmatrix<f16x32x32,matrix_a>
            %tb = cooperative_matrix_load %bb[%k,%j0] : coopmatrix<f16x32x32,matrix_b>
            %r  = cooperative_matrix_mul_add %ta, %tb, %cc : coopmatrix<f32x32x32,matrix_acc>
            yield (%r)
        }
        cooperative_matrix_store %acc, %c[%i0,%j0]
    }
}
```

---

## 4. Calling convention (how `set_args` must look)

Each tensor-language argument expands to one or more kernel arguments, **in parameter order**:

| Argument type | Kernel arguments |
|---|---|
| scalar `iN`/`fN`/`index` | the scalar (`index` → `int64_t`); `c32`/`c64` → 2 floats/doubles (`std::complex<T>`); `bool` args are not allowed |
| `memref<…>` | base pointer `T*`, then one `int64_t` per `?` in the **shape** (left→right), then one per `?` in the **stride** (left→right) |
| `group<memref<…>x?>` | `T**`, then one `int64_t*` *array* per `?` in the memref shape/stride, then group size (if `?`), then offset (if `offset: ?`) |

Canonical strides count: `memref<f32x5x?x6>` has stride `strided<1,5,?>`, so it takes
`(ptr, shape1, stride2)`.

Examples:
```
func @k(%a: memref<f32x56x9x?>)                       -> set_args(a_ptr, n)
func @k(%A: group<memref<f32x9x9>x?>)                 -> set_args(A_ptrs, howmany)
func @k(%Q: memref<f16x64x?x?x?> {stride_gcd=[1,64,64,64]})
   ; strided<1,64,?,?>  -> set_args(Q, s1, s2, s3, stride2, stride3)
```

**Alignment:** the base pointer must satisfy the default alignment (128 B for Intel core infos, or
`CL_DEVICE_MEM_BASE_ADDR_ALIGN` via OpenCL) unless the argument has an `alignment=` attribute. Passing
`ptr + offset` as a base pointer is the classic way to get wrong results — either add `{alignment=…}`
or call `tinytc::set_default_alignment(info.get(), bytes)` before compiling.

---

## 5. SYCL API (C++, `#include <tinytc/tinytc_sycl.hpp>`)

Only Intel GPUs on the **Level Zero** and **OpenCL** SYCL backends are supported (the SYCL layer dispatches
on `dev.get_backend()` and builds native modules). CUDA/HIP backends will not work.

| Function | Purpose |
|---|---|
| `get_support_level(const sycl::device&)` → `support_level::{none,basic,tuned}` | check before use |
| `create_core_info(const sycl::device&)` → `shared_handle<tinytc_core_info_t>` | device properties for codegen |
| `create_kernel_bundle(ctx, dev, tinytc_prog_t, core_features = 0)` | JIT a parsed/built program |
| `create_kernel_bundle(ctx, dev, const_tinytc_binary_t)` | load a precompiled SPIR-V binary |
| `create_kernel(bundle, "name")` → `sycl::kernel` | |
| `get_group_size(kernel)` → `sycl::range<3>` | work-group size chosen by compiler |
| `get_execution_range(kernel, sycl::range<3>{nz, ny, nx})` → `sycl::nd_range<3>` | **ZYX order!** |
| `get_global_size(num_groups, local_size)` | |
| `create_recipe_handler(queue, recipe)` / `(ctx, dev, recipe)` | recipes |
| `tinytc::submit(handler, queue[, deps])` → `sycl::event`; `tinytc::parallel_for(handler, cgh)` | launch recipe |

### 5.1 JIT from text — canonical host program
```cpp
#include <sycl/sycl.hpp>
#include <tinytc/tinytc.hpp>
#include <tinytc/tinytc_sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    try {
        if (tinytc::get_support_level(q.get_device()) < tinytc::support_level::basic) {
            std::cerr << "device not supported by tinytc\n";
            return 1;
        }
        auto ctx = tinytc::create_compiler_context();
        tinytc::set_error_reporter(ctx.get(), [](char const *what, const tinytc_location_t *, void *) {
            std::cerr << what << std::endl;
        });
        // tinytc::set_optimization_level(ctx.get(), 2);
        // tinytc::set_optimization_flag(ctx.get(), tinytc::optflag::unsafe_fp_math, 1);

        auto prog = tinytc::parse_file("fused_kernel.ir", ctx.get());   // or parse_string(src, ctx.get())
        auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), prog.get());
        auto kernel = tinytc::create_kernel(bundle, "fused_kernel");

        std::int64_t howmany = 1000;
        float *K = sycl::malloc_device<float>(56 * 56, q);
        float *P = sycl::malloc_device<float>(56 * 9 * howmany, q);
        float *Q = sycl::malloc_device<float>(56 * 9 * howmany, q);
        float **A = sycl::malloc_device<float *>(howmany, q);   // array of device pointers
        // ... fill ...

        auto range = tinytc::get_execution_range(kernel, sycl::range<3u>{1, 1, std::size_t(howmany)});
        for (int step = 0; step < 100; ++step) {                     // reuse the compiled kernel
            q.submit([&](sycl::handler &h) {
                h.set_args(K, P, howmany, A, howmany, Q, howmany);   // calling convention, §4
                h.parallel_for(range, kernel);
            }).wait();
        }
    } catch (tinytc::status const &st) {
        std::cerr << "Error (" << static_cast<int>(st) << "): " << tinytc::to_string(st) << std::endl;
        return 1;
    } catch (std::exception const &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
```
Pass `tinytc_core_feature_flag_large_register_file` as the 4th argument of `create_kernel_bundle` to
request the large GRF mode.

### 5.2 Compile to binary in-process (e.g. to cache or inspect)
```cpp
auto info = tinytc::create_core_info(q.get_device());           // or create_core_info_intel_from_arch(tinytc::intel_gpu_architecture::pvc)
tinytc::set_core_features(info.get(), tinytc_core_feature_flag_large_register_file);
auto bin = tinytc::compile_to_spirv_and_assemble(prog.get(), info.get());   // modifies prog (runs passes)
auto raw = tinytc::get_raw(bin.get());                          // raw.data, raw.data_size
auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), bin.get());
```
`compile_to_spirv(prog, info)` returns a SPIR-V module you can `print_to_string`/`dump`, then
`spirv_assemble(mod)`.

### 5.3 Load SPIR-V produced by the `tinytc` CLI
```cpp
std::vector<std::uint8_t> spv = read_file("kernel.spv");
auto ctx = tinytc::create_compiler_context();
auto bin = tinytc::create_binary(ctx.get(), tinytc::bundle_format::spirv, spv.size(), spv.data(),
                                 /*core_features=*/0);   // same -F flags as used offline
auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), bin.get());
```
Offline binaries are specialised for the `-d` arch (subgroup size, register budget); compile for the
device you run on.

### 5.4 Recipes (pre-built generators)
Small batched GEMM — `C_i = α op(A_i) op(B_i) + β C_i`, fixed shapes and strides; kernels `gemm` and
`gemm_beta0`:
```cpp
auto info = tinytc::create_core_info(q.get_device());
auto ctx  = tinytc::create_compiler_context();
auto rec  = tinytc::create_small_gemm_batched(info.get(), tinytc::to_type<float>(ctx.get()),
                tinytc::transpose::N, tinytc::transpose::N, M, N, K,
                /*ldA*/M, /*strideA*/M*K, /*ldB*/K, /*strideB*/K*N, /*ldC*/M, /*strideC*/M*N);
auto h = tinytc::create_recipe_handler(q, rec.get());
tinytc::set_small_gemm_batched_args(h.get(), howmany, 1.0f, A, B, 0.0f, C);   // USM ptrs auto → mem
tinytc::submit(h.get(), q).wait();
```
Tall-and-skinny GEMM (`M ≫ N, K`): `create_tall_and_skinny(info, ty, N, K, M_block_size=0)` or
`create_tall_and_skinny_specialized(info, ty, M, N, K, ldA, ldB, ldC, alignA, alignB, alignC, M_block_size)`
(use `tinytc::dynamic` for runtime values), then
`set_tall_and_skinny_args(h.get(), M, alpha, A, ldA, B, ldB, beta, C, ldC)`.
`get_prog(rec.get())` returns the generated IR — `dump()` it to learn idioms.
Scalar types passed to `set_*_args` must match the recipe's element type exactly.

---

## 6. Core C++ API notes
- Handles: `shared_handle<T>` (ref-counted) and `unique_handle<T>`; default-constructed handles are null.
  Always use `create_*` functions. Pass raw handles with `.get()`.
- Errors: the C++ API throws `tinytc::status` (print with `tinytc::to_string`); the builder throws
  `tinytc::builder_error`. The C API returns `tinytc_status_t`. Without an error reporter on the compiler
  context, parse errors are silent — always install one.
- Core info without a device: `create_core_info_intel_from_arch(intel_gpu_architecture::bmg)`,
  `create_core_info_intel_from_name("pvc")`, or `create_core_info_generic(register_space, max_wg_size, sgs…)`.
  Query: `get_subgroup_sizes`, `get_register_space`, `have_spirv_feature`, `get_default_alignment`.
- Passes from code: `run_function_pass("insert-barrier", prog.get(), info.get())`,
  `list_function_passes(n, names)`, `print_to_string(prog.get())`.
- Builder API (`tinytc/builder.hpp`): build IR programmatically when `$`-templating is not enough
  (`get<f32_type>(ctx)`, `get<memref_type>(…)`, `create_func`, `region_builder{body}.create<gemm_inst>(…)`,
  `create_prog`, `add_function`). See `examples/builder/main.cpp`. For most templating needs, prefer
  generating text with `$` variables and `parse_string` — it's shorter and gives located errors.
- Level Zero / OpenCL equivalents exist (`tinytc_ze_kernel_bundle_create_with_binary`,
  `tinytc_ze_get_group_count(howmany)`, `tinytc_cl_get_group_size/global_size`), see `examples/simple_ze`,
  `examples/simple_cl`.

---

## 7. Performance guidance
- Make sizes static wherever possible (`memref<f32x56x9>` beats `memref<f32x?x?>`); generate specialised
  variants with `$` variables. Use `shape_gcd`/`stride_gcd`/`alignment` when sizes must stay dynamic.
- Fuse: chain several linalg ops in one kernel with `alloca` temporaries in SLM; the compiler reuses SLM
  across non-overlapping lifetimes and inserts barriers (between linalg ops only; see §8).
- **Layout decides more than tuning.** In batched tile kernels put the batch (element) index fastest, in
  blocks of B elements (`memref<f32xB x…x?>`, one block per work-group), so tile rows load
  contiguously. The same fused kernel on element-contiguous (AoS) arrays ran at 0.69–0.80× of the
  blocked one (CEED BK3 on an Arc A770).
- **Sub-group size sets the minimum block.** Tile rows (A and C) must be a multiple of the sub-group
  size, so with the batch index as the row index a block holds ≥ 8 or 16 elements, and SLM grows with
  it. When SLM is the limit, an explicit `subgroup_size=8` (on devices that have it, e.g. DG2, TGL)
  halves the block and lets larger problems fit; it also won outright there for mid-size tiles.
- **Odd contraction lengths:** for K = 5, 7, … store the small right-hand matrix zero-padded to a
  power of two (`memref<f32xK x N,strided<1,$K_pad>>`) and load the left-hand tile with
  `cooperative_matrix_load.cols_checked` (out-of-range columns read 0), so the padded K costs no
  bounds logic. `fuse` adjacent modes to make the tile's row index (e.g. element × spatial index).
- For SLM-bound kernels, check the footprint before building (`set-stack-ptr`, §2) and reject
  configurations on the host; see §8 for the error you get otherwise.
- **When the work arrays do not fit in SLM, spill one to global memory** instead of shrinking the block
  below the sub-group size. Add a scratch argument with one slice per work-group and select it with
  the group id; its views drop `,local`:
  ```
  %Sg: memref<f32x$B x$Q x$Q x$Q x?>                        ; argument, one slice per work-group
  %S = subview %Sg[0:$B,0:$Q,0:$Q,0:$Q,%gid] : memref<f32x$B x$Q x$Q x$Q>
  ```
  Between steps that communicate through it, use `barrier.global.local` (fences both spaces), not
  `barrier.local`. The slice is private to the work-group, so no cross-group sync is needed. In CEED
  BK3 on an A770 this cost ~10% where SLM would have fit, and at p = 7, 8 (where it would not) it ran
  at 0.88–0.92× of hand-written OpenMP/SYCL kernels.
- Let the compiler pick `subgroup_size`/`work_group_size` first; inspect with
  `tinytc-opt -pwork-group-size`, then override if needed.
- For low precision (f16/bf16 inputs, f32 accumulation) on PVC/BMG, GEMMs map to XMX via cooperative
  matrices; write explicit `cooperative_matrix_*` code for attention-style kernels (see `examples/flash_attention`).
- `-F large-register-file` helps register-heavy kernels on PVC.
- The flash-attention example recommends `IGC_EnableSelectiveScalarizer=1` and
  `NEO_CACHE_PERSISTENT=0` for best performance/benchmarking.
- Time with many repetitions after a warm-up launch; never include JIT in timings.

---

## 8. Gotchas and doc errata (checked against v0.5 sources)
- **No `:` full slice.** `subview %X[:,%j]` is a parse error; write `%X[%c0:%m, %j]` with `%m = size %X[0]`.
- Slices are **offset:size**, not begin:end. A single index (or size 0) drops the mode.
- No immediates: `add %x, 1` is invalid — use `%c1 = constant 1 : i32`.
- Types must match exactly in arithmetic; use `cast` (`subgroup_id` is `i32`, loop indices are usually `index`).
- `group_id`/`num_groups` give `index`; `subgroup_id`, `num_subgroups`, `subgroup_size` give `i32`.
- `get_execution_range` takes **{z, y, x}**; `group_id.x` corresponds to the *last* entry.
- Docs typos to ignore: the tutorial's `set_error_reporter([](ctx.get(), …), nullptr)` is wrong — correct form is
  `set_error_reporter(ctx.get(), lambda)`; the core guide's `handler.submit(queue)` for recipes is really
  `tinytc::submit(handler.get(), queue)`; the `cooperative_matrix_apply` example's `cmp.le` is
  `less_than_equal` in the real grammar. The printer shows `cooperative_matrix_apply (%i,%j,%v) in %t`;
  the parser also accepts `=%t`.
- Barriers are automatic only between linalg instructions in collective regions; inside `parallel {}`,
  and between consecutive `foreach`/`foreach_tile` loops that communicate through SLM, you must place
  `barrier.local` yourself. Missing ones give wrong results, not errors.
- **SLM over the per-work-group limit (64 KiB on DG2) is not diagnosed by tinytc.** Kernel creation
  fails with only `tinytc error 18: Error occured in compute runtime`; underneath, `zeModuleCreate`
  returns `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` with an empty build log, although device memory is
  free. Size the allocas (see `set-stack-ptr`) against the device limit before compiling.
- **Multi-dimensional `foreach_tile` parallelizes only its first two dimensions:** dimension 0 is
  spread over `subgroup_id.x`, dimension 1 over `subgroup_id.y` (1 sub-group with
  `work_group_size=[n,1]`), the rest run serially in each sub-group. A loop like
  `foreach_tile (%b,%q,%p)=…,(%cB,%cQ,%cQ) as …<=($M,1,1)` with B ≤ M leaves one sub-group doing
  all the work (8–30× slower in BK3). Flatten to 1-D instead and derive the indices:
  `foreach_tile (%sM)=(%c0),(%cNS_times_M) as (%n)<=($M)` with `%s = div %sM, %cM`. Tile shape 0
  must be a multiple of the sub-group size (`Tile shape 0 is not a multiple of the subgroup size`),
  so step by $M rather than 1.
- `rows_checked`/`both_checked` loads fall back from sub-group block IO to per-lane loads
  (`coopmatrix_impl_block.cpp` accepts only unchecked or `cols_checked`). Measured cost in BK3:
  0–8%, so the checks themselves are cheap; idle rows are what hurt.
- `$` variables may be defined anywhere, including inside a function body; a type alias such as
  `$mat_t = coopmatrix<f32x$M x$K,matrix_acc>` shortens repeated tile types.
- f64, bf16, f16 and float atomics depend on SPIR-V capabilities of the device (check with
  `tinytc-ze-info`/`tinytc-cl-info`); integrated GPUs often lack fp64.
- `compile_to_spirv*` mutates the program (passes run in place). Re-parse if you need the original.

## 9. Where to look in the repo
- `docs/manual/tensor-ir.rst` — full language reference (grammar + restrictions for every instruction).
- `docs/manual/calling_convention.rst`, `core.rst`, `builder.rst`, `preprocessor.rst`, `tutorial_matrix_chain.rst`.
- `include/tinytc/tinytc_sycl.hpp`, `core.hpp`, `builder.hpp.mochi` (generated into `builder.hpp`) — authoritative APIs.
- `examples/`: `matrix_chain`, `tall_and_skinny`, `flash_attention` (SYCL + preprocessor template),
  `benchmark`, `builder`, `simple_ze`, `simple_cl`, `jit`.
- `test/lit/spv/*.ir`, `test/lit/opt/*.ir` — small, known-good snippets of every instruction.