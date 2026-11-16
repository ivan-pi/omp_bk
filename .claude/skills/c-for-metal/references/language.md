# C for Metal language reference (condensed)

Condensed from the *C for Metal Language Specification* (cm-compiler,
`clang/compute_sdk/docs/cmlangspec/cmlangspec.rst`) and the CGO 2021 paper
(Lueh et al., arXiv:2101.11049). Section numbers refer to the spec. Memory
access is in `memory.md`.

## Contents
1. Execution model
2. Types
3. Base objects and references
4. Operators, conversions, saturation
5. Select family: (), [], select, select_all, row/column, iselect, replicate
6. format, merge, masks, any/all
7. Control flow: scalar and SIMD
8. Functions, kernels, globals, specialization constants
9. Built-in functions (by category)
10. Float control, pragmas, printf
11. Restrictions (what C++ you cannot use)
12. Worked idioms

---

## 1. Execution model

- A kernel (`_GENX_MAIN_`) describes **one hardware thread**, not one work-item.
  All SIMD parallelism inside the thread is explicit: an operation on a
  `vector<float, 32>` is a SIMD32 operation; the compiler splits it into
  instructions of legal width (SIMD8/16/32 depending on type and target).
- Variables of `vector`/`matrix` type live in the **general register file
  (GRF)**. They cannot have their address taken. GRF per thread: 128 x 32 B
  = 4 KB on Gen9..XeHPG by default (64 B registers on XeHPC/Xe2+); 256
  registers with `-Qxcm_doubleGRF` / `-double-GRF` / large-GRF mode (halves
  the threads per EU). A spill is usually a performance disaster, so size the
  working set to the GRF.
- Thread identity: `cm_group_id(d)`, `cm_local_id(d)`, `cm_local_size(d)`,
  `cm_group_count(d)`, plus linear forms `cm_linear_global_id()`,
  `cm_linear_local_id()`, `cm_linear_group_id()` ... (spec 4.7). With the
  OpenCL/L0 runtimes the NDRange counts **threads** (SIMD1 dispatch):
  global size = number of CM threads, local size = threads per group.
- Threads of one group share SLM and can `cm_barrier()`; nothing else is
  shared except global memory.

## 2. Types (spec 2.1)

| Type | Notes |
|---|---|
| `char uchar short ushort int uint float half` | `half` is IEEE binary16 |
| `double` | operators except `/`; intrinsics only `cm_abs/min/max/add/mul/sum`; target must define `CM_HAS_DOUBLE` (DG2 has no FP64 hardware: it is emulated or rejected) |
| `long long`, `unsigned long long` | 64-bit ints; no `*`, `/`, no intrinsics (emulated on many targets) |
| `svmptr_t` | integer of pointer size for SVM/stateless addresses (byte addressing) |
| `bfloat16`, `tfloat32`, `bfloat8`, `hfloat8` | target dependent (`CM_HAS_BF16` ...) |
| `SurfaceIndex`, `SamplerIndex` | opaque handles; only as kernel parameters (or `vector<SurfaceIndex, N>` parameter) |

Masks: `int` (packed bits), `vector<ushort, N>`, `matrix<ushort, R, C>`.
In `merge` only the LSB of a mask element counts; as a SIMD control-flow
condition any non-zero element is true.

## 3. Base objects and references (spec 2.2)

```cpp
vector<short, 8> v;           // 8 shorts in registers
matrix<int, 4, 8> m;          // 4 rows x 8 columns, row-major
vector_ref<int, 8> r = m.row(2);           // alias, no storage
matrix_ref<char, 4, 32> b = m.format<char, 4, 32>();
```

- Sizes and element types are compile-time constants. Total size must fit
  the GRF.
- `vector_ref`/`matrix_ref` are views into a base object (regioned access, no
  copy). They cannot be struct fields or function return types; they are the
  only pass-by-reference parameter kind.
- Structs may contain scalars and vector/matrix members (properly aligned),
  not references.
- Query sizes: `m.n_rows()`, `m.n_cols()`, `v.n_elems()`.

Initialization (spec 3.2.2-3.2.3):

```cpp
const short init[8] = {0, 1, 2, 3, 4, 5, 6, 7};   // global const array
vector<ushort, 8> v(init);                         // compiler emits vector immediates
vector<uchar, 2> s = vector<uchar, 2>(vi, SAT);    // saturating conversion constructor
#include <cm/cmtl.h>
cm_vector(v2, ushort, 16, 2, 3);   // declares v2 = {2, 5, 8, ...}
cm_matrix(m2, ushort, 4, 8, 10, 5);
cmtl::cm_vector_assign(v3.select<10, 1>(2), i, 3); // runtime start value
```
(4-element `cm_vector`s compile with the current cmc; the 2021 `cmtl.h`
rejects them, `legacy-sdk.md`.)

## 4. Operators, conversions, saturation

- `= + - * / % += -= *= << >> & | ^ ~ !` are component-wise. A scalar operand
  is broadcast. Operands may differ in shape if element counts match (copied
  row-major).
- Comparisons `< <= > >= == !=` return `vector<ushort, N>` (0/1) masks.
- `&&` and `||` take scalar (contextually bool) operands.
- Result element type follows C++ promotion; assign to a typed object to
  convert (`f = i;`) or use an explicit constructor `vector<short, 8>(i)`.
  float -> integer conversion truncates toward zero.
- Saturation: constructor flag `SAT`, intrinsic flag `SAT`/`_GENX_SAT`.
  Float saturation clamps to [0, 1]; integer saturation clamps to the
  destination range.
- Evaluation order (3.3): all operands are read before the result is
  written, so `m.select<2,1,4,1>(0,0) += m.select<2,1,4,1>(1,0)` uses the old
  values of overlapping rows.

## 5. The select family (spec 3.2.9)

All except `iselect` and `replicate` return references (usable as l-values).

| Form | Meaning |
|---|---|
| `v(i)`, `v[i]`, `m(i, j)`, `m[i][j]` | element |
| `v.select<size, stride>(i = 0)` | `size` elements starting at `i`, `stride` apart |
| `m.select<vsize, vstride, hsize, hstride>(i = 0, j = 0)` | sub-matrix: `vsize` rows `vstride` apart, `hsize` columns `hstride` apart, origin (i, j) |
| `x.select_all()` | the whole object as a reference |
| `m.row(i)`, `m.column(j)` | row / column reference (column is a strided region) |
| `v.iselect(idx)` | **r-value** gather from a register vector, `idx` is `vector<ushort, K>` of element indices; result has K elements (register-indirect addressing) |
| `m.iselect(rows, cols)` | gather of matrix elements |
| `x.replicate<REP>()` | REP copies of the whole object |
| `x.replicate<REP, W>(i)` | REP copies of W consecutive elements |
| `x.replicate<REP, VS, W>(i)` | REP blocks of W consecutive elements, blocks VS apart |
| `x.replicate<REP, VS, W, HS>(i)` | REP blocks of W elements HS apart, blocks VS apart |

Rules: strides must be > 0, and a stride must be 1 when its size is 1.
Offsets may be runtime values. A constant offset compiles to direct
register regioning and a variable one to indirect addressing, which is
slower and narrower. `replicate` and `select` map straight onto Gen region
syntax `<VS;W,HS>`, so they compile to plain `mov`s or fold into the
consuming instruction's source operand. That is CM's main performance tool:
data shuffles that SIMT code does with shuffles or SLM cost nothing here.

Out-of-range selects are allowed only when the out-of-range elements don't
affect the result (the values are undefined).

## 6. format, merge, masks, reductions (3.2.10-3.2.12)

```cpp
matrix<int, 4, 4> m1;
matrix_ref<char, 4, 16> bytes = m1.format<char, 4, 16>(); // reinterpret bits
vector_ref<float, 16>   flat  = m4.format<float>();        // as 1D
```
`format` needs a contiguous, aligned source. The result may be smaller than
the source but not larger.

```cpp
v.merge(x, mask);        // v[i] = mask[i] ? x[i] : v[i]       (predicated mov)
v.merge(x, y, mask);     // v[i] = mask[i] ? x[i] : y[i]       (sel)
v.merge(x, 0b0101);      // int mask: bit i selects element i
ushort a = (m > 0).any(); // 1 if any element true
ushort b = mask.all();
```

## 7. Control flow (spec 3.4)

**Scalar** C++ `if/for/while/switch/break/continue/return` with a scalar
condition applies to all lanes, so the branch is uniform. Use `.any()` or
`.all()` to turn a mask into a scalar condition.

**SIMD control flow** (per-lane divergence with hardware masks, `goto`/`join`):

```cpp
vector<float, 32> x, y;
SIMD_IF_BEGIN(x < 0.0f) {
  y = 0.0f;
} SIMD_ELSEIF(x > 10.0f) {
  y = x * 0.5f;
} SIMD_ELSE {
  y = x + 1.0f;
} SIMD_IF_END;

SIMD_DO_WHILE_BEGIN {
  ...
  SIMD_IF_BEGIN(v < 0) { SIMD_BREAK; } SIMD_IF_END;
} SIMD_DO_WHILE_END(cond < 32);
```

Rules:
- The SIMD width is set by the condition: a power of two from 2 to 32. Nested
  blocks must have the same width.
- Every statement inside must have that width or width 1. A scalar
  statement runs if any lane is active.
- Not allowed inside: scalar control flow, other jump statements,
  user-defined function calls, block reads and writes, sampler and media
  block I/O, and dot-product/SAD/line intrinsics. Scattered read/write is
  allowed when its width equals the block width.
- Reductions (`any`, `all`, `cm_sum`, ...) inside a block act on the active
  lanes and must have the block width.
- An alternative with no branches is to compute both sides and `merge`.
  That is usually better for short bodies; SIMD control flow pays off when
  the skipped work is large.

## 8. Functions, kernels, globals (spec 2.4, 4.1-4.2)

```cpp
extern "C" _GENX_MAIN_ void kname(SurfaceIndex buf [[type("buffer_t")]], uint n) {...}
_GENX_ inline float helper(vector<float, 16> v) { return cm_sum<float>(v); }
_GENX_ void out_param(matrix_ref<float, 2, 2> m) { m = 1.0f; }  // by reference
```

- Host programs find a kernel by its plain name (`clCreateKernel`,
  `ext_oneapi_get_kernel`). That name comes from the `!genx.kernels`
  metadata, so even without `extern "C"` the SPIR-V entry point is
  unmangled (tested with both cmc). `extern "C"` is still the convention:
  it also keeps the LLVM IR symbol and the asm dump names readable, and it
  forbids accidental overloads.

- **Bind `constexpr`-function results to `constexpr` variables.**
  `v.select<32, 1>(off_of(1))` or `read(x, off_of(1), ...)` with
  `constexpr int off_of(int)` is evaluated at run time, since nothing
  forces a constant there. The asm then contains a subroutine `call`/`ret`
  and the select becomes indirect `a0` addressing: 48 instructions with 8
  indirect accesses on DG2, against 25 and none for
  `constexpr int o1 = off_of(1); v.select<32, 1>(o1)`. Both cmc versions
  behave this way (tested).

### What `_GENX_MAIN_`, `_GENX_` and friends are

Macros from `cm/cm_common.h` (identical in the 2021 and current headers)
for clang `__declspec` attributes that the CM front end understands. Spec
§4.1 "Function Qualifiers" defines them. What they turn into
(`cmc -E`, `-emit-llvm -S`, SPIR-V text):

| Macro | Expands to | Effect | In LLVM IR / SPIR-V |
|---|---|---|---|
| `_GENX_MAIN_` | `__declspec(genx_main)` | a **kernel** (entry point). It can't be called from another kernel; a file may hold several | function attribute `"CMGenxMain"`, an entry in `!genx.kernels` (plain name, argument kinds, `buffer_t`/`svmptr_t`/... annotations) → SPIR-V `OpEntryPoint Kernel` + `VectorComputeFunctionINTEL` |
| `_GENX_` | `__declspec(genx)` | marks a device function. **Optional**: an unqualified C++ function called from a kernel compiles the same | nothing distinctive: `internal` function (identical to an unqualified one) |
| `inline` (standard C++) | | the compiler inlines it, with a warning or error if it can't. This, not `_GENX_`, decides inlining | the function disappears from the IR |
| `_GENX_STACKCALL_` | `__declspec(genx_stackcall)` | a real call with a stack frame instead of inlining or a `call`/`ret` subroutine (for code size or indirect calls) | attribute `"CMStackCall"` → SPIR-V `StackCallINTEL` |
| `_GENX_VOLATILE_`, `_GENX_VOLATILE_BINDING_(off)` | `__declspec(genx_volatile[(off)])` | a global kept in memory-like form (less register pressure), or pinned at a GRF byte offset | |
| `_SIMT8_`, `_SIMT16_`, `_SIMT32_` | `__declspec(genx_SIMT(N))` | SIMT entry functions (IR `"CMGenxSIMT"="N"`), not a kernel dispatch width | on a `_GENX_MAIN_` kernel both IGC 2.42 and 2.11 crash (tested) |
| `_GENX_ROUNDING_MODE_(x)` | `__declspec(cm_float_control(x))` | the kernel's float rounding/denormal control | |
| `_CM_CALLABLE_`, `_CM_ENTRY_`, `_CM_INPUT_`, `_CM_OUTPUT_` | `__declspec(cm_callable)` ... | CM-runtime "callable kernel" interfaces (legacy media pipelines) | |

**The name:** "GenX" is how CM and IGC refer to the target. The LLVM
target triple is `genx64-unknown-unknown`, the intrinsics are
`llvm.genx.*`, and the IGC backend passes are "GenX". Intel called its GPU
architectures and their ISA "Gen" (Gen7.5, Gen9, Gen11, Gen12), and
`CM_GENX` holds the generation times 100 (Gen9 = 900, Gen12 = 1200). So
"GenX" most likely means "Gen *X*", any Gen generation. No document we
have spells this out, so treat it as an inference; nothing supports
"X = executable".
- A kernel and every function it calls must be in the same file.
  Recursion is not allowed. Parameters can be scalars, vector/matrix
  (by value), `vector_ref`/`matrix_ref` (by reference), or SurfaceIndex.
- Pass by reference means copy-in/copy-out: a written reference parameter must
  not alias another reference parameter or a global used in the function.
  Reference accesses use indirect addressing (SIMD8-limited), so pass small
  inputs (2 GRFs or less) by value.
- Globals are **thread-private** registers shared by the kernel and its
  callees. They cannot be initialized. `_GENX_VOLATILE_` restricts
  optimization to lower register pressure, and `_GENX_VOLATILE_BINDING_(off)`
  pins the global at a GRF byte offset.

### Specialization constants, and the alternatives

CM has SPIR-V specialization constants. They aren't in the language spec,
only in `cm/cm_spec_constant.h` (included by `cm/cm.h`) and the upstream
front-end tests (`clang/test/CMFE/cm_spec_constant/`):

```cpp
const int   blocks = cm_spec_constant<int, 0, 4>(); // integral: SpecId 0, default 4
const float s      = cm_spec_constant<float, 1>();  // floating: SpecId 1, default ALWAYS 0
```

The call becomes `OpSpecConstant` with that SpecId. The value is set when
the SPIR-V is finalized:
- OpenCL: `cl::Program::setSpecializationConstant(id, value)`
  (`clSetProgramSpecializationConstant`) between `cl::Program(ctx, il)` and
  `build()`;
- Level Zero: `ze_module_constants_t` in `ze_module_desc_t::pConstants`;
- offline: `ocloc compile ... -spec_const file`, one `id: value` per line.

IGC then folds the value like a literal, so branches on it vanish and loops
get constant trip counts. It is **not a C++ constant expression**: it can't
size a `vector`, be a template argument, or drive `if constexpr` or
`select<N,S>`. Size registers for a compile-time maximum and branch on the
constant (`examples/specconst_genx.cpp`).

**Requirements and results:**
- `intel-igc-cm` cmc (1.0.225) with its matched IGC (`pipeline.md` §7):
  `blocks=8 s=2.5` and `blocks=3 s=-1.5` PASS (tested on an A770 and a UHD 750).
  (The 2021 SDK cmc crashes on it: `legacy-sdk.md`.)
- **Binaries:** a `.bin` built without values carries the defaults, and it
  loads on a newer driver too.
- **`ocloc -spec_const`:** ocloc 25.18 (the release matched to
  `intel-igc-cm` 1.0.225) lacks it, while a newer ocloc such as 26.40 has
  it but can't read that SPIR-V. So set non-default values at run
  time.

Without a current toolchain, use these instead, the CM forms of OpenCL's `-D`
at `clBuildProgram` time:

| Need | CM way | Cost |
|---|---|---|
| shapes, widths, unroll counts, register budget | `-D` at cmc time (`CMC_FLAGS=-DCHUNK=...`) or templates | one SPIR-V / binary per variant, built offline. The run-time `-cmc` source path (`clBuildProgram` with `-cmc -D...`, OpenCL style) works only with a current cmc and its matched runtime |
| several variants in one binary | `template <int N> _GENX_ inline void body(...)` plus one `extern "C" _GENX_MAIN_` wrapper per instantiation (`kern_8`, `kern_16`); the host picks a kernel by name | code size; each variant compiles once |
| a value that only steers control flow or arithmetic | a scalar kernel argument | almost free: it is uniform, a scalar in a register, and `if (arg)` is a scalar branch, not SIMD control flow. It isn't folded, so loops on it are not unrolled |

In SYCL ESIMD the spec-constant API is SYCL's own (`specialization_id`,
`kernel_handler::get_specialization_constant`), with upstream tests in
`sycl/test-e2e/ESIMD/spec_const/` (bool to double).

## 9. Built-ins (spec 4.3; `T` = result type, most take an optional `SAT` flag)

- Arithmetic: `cm_abs cm_add cm_mul cm_div cm_quot cm_mod cm_avg cm_min cm_max cm_imul cm_addc cm_subb`
- Rounding: `cm_rndd cm_rndu cm_rnde cm_rndz cm_frc`
- Reductions: `cm_sum<T> cm_prod<T> cm_reduced_min<T> cm_reduced_max<T>`
- Dot products: `cm_dp2 cm_dp3 cm_dp4 cm_dph cm_dp4a cm_line cm_lrp cm_pln cm_sad2 cm_sada2`
- Math (hardware precision unless noted): `cm_inv cm_log cm_exp cm_sqrt cm_rsqrt cm_pow cm_sin cm_cos cm_sincos cm_asin cm_acos cm_atan cm_tanh cm_sigmoid`. IEEE-exact: `cm_div_ieee cm_sqrt_ieee` (`CM_HAS_IEEE_DIV_SQRT`).
- Bits: `cm_shl cm_shr cm_asr cm_lsr cm_rol cm_ror cm_cbit cm_fbl cm_fbh cm_lzd cm_bf_insert cm_bf_extract cm_bf_reverse cm_bfn` (`cm_bfn<expr>(a,b,c)`: any 3-input boolean function in one instruction)
- Masks: `cm_pack_mask(vector<ushort,N>) -> uint`, `cm_unpack_mask<T,N>(uint)`
- Conversions: `cm_bf_cvt cm_tf32_cvt cm_bf8_cvt cm_hf8_cvt cm_srnd`, `cm_f32tof16`
- XMX: `cm_dpas cm_dpasw cm_bdpas` (spec 4.20, see the dpas section below)
- Time: `cm_clock cm_rdtsc`. Misc: `cm_get_hwid cm_assert`

### XMX / DPAS

`D(MxN) = A(MxK) * B(KxN) + C`.
`cm_dpas<Src1Prec, Src2Prec, SystolicDepth=8, RepeatCount=M>(C, B_vnni, A)`.

- N = ExecSize: **8 on XeHPG (DG2/Arc A)**, 16 on XeHPC (PVC) and Xe2/Xe3.
- K = 8 * OpsPerChannel: 2 for hf/bf, 4 for int8, 8 for int4/int2, 1 for tf32.
- A is row-major packed into `int`/`uint`, Src2Size = M*K*bits/32.
- B is VNNI-packed: each dword holds 32/bits consecutive K-elements of one
  column, Src1Size = K*N*bits/32.
- C/D: `vector<float or int, M*N>` row-major.
- Precision names: `CM_PRECISION_HF BF U8 S8 U4 S4 U2 S2 TF32 ...`.
- `cmc -march=dg2` gives a native `dpas.8x8` (`examples/dpas_genx.cpp`);
  `CM_HAS_DPAS` tells whether the target has XMX (`targets.md`).

## 10. Float control, pragmas, printf

```cpp
_GENX_MAIN_ _GENX_FLOAT_CONTROL_(CM_RTZ | CM_DENORM_F_ALLOW) void k(...) {...}
```
Rounding is `CM_RTE` (default), `CM_RTP`, `CM_RTN` or `CM_RTZ`. Denormals are
`CM_DENORM_RTZ` (default: flush) or `CM_DENORM_{D,F,HF}_ALLOW` /
`CM_DENORM_ALLOW`. `CM_FLOAT_MODE_IEEE` is the default single-precision mode.
`_GENX_ROUNDING_MODE_` is an older synonym.

`#pragma unroll` / `#pragma unroll(n)` before loops. Fully unroll loops whose
counter feeds a `select` offset or a `row(i)` index, so the offsets become
constants and the regions stay direct.

`printf(fmt, ...)` in-kernel: format string under 128 bytes, at most 64
arguments, no `%p` or `%n`. It needs a printf buffer from the host runtime.
The cmc option `-cm-printf-spec=ocl` selects the OpenCL printf protocol
(the alternative is the legacy CM-runtime one); this is untested.
Treat printf as a debugging aid.

## 11. Restrictions (spec 2.6)

Kernel-side code cannot use: pointers (except the newer address-space
pointer support of spec chapter 7 / `-fcm-pointer`), C++ references, classes
other than structs and the CM types, inheritance, arrays (use vectors; const
global arrays only as initializers), exceptions, RTTI, dynamic allocation,
static or volatile locals, or calls to external functions. Templates,
`constexpr`, `static_assert`, `if constexpr` and lambdas inside a function do
work, because the front end is clang.

**C++ level:** cmc compiles C++17 (`__cplusplus` = 201703L) and
rejects `-std=`. Variadic templates, `decltype`, variable templates, fold
expressions and `if constexpr` all work, so compile-time configuration like
the cuSolverDx/cuBLASDx descriptors works too:
`using Solver = decltype(Size<16,16,4>() + Precision<float>() + Function<posv>() + ...)`,
with traits read from the composed type (`examples/traits_genx.cpp`, tested on
an A770). Empty "operator" structs combined by inheritance and used only at
compile time compiled fine, despite the inheritance restriction above.
Give each operator a uniquely named member: two bases that both define
`value` make `D::value` ambiguous. Only a minimal `std` exists in kernels
(`std::is_same` and a few other traits, but no `is_same_v` and no standard
library), so write `std::is_same<A, B>::value`.
Two template pitfalls seen in `examples/gesv_genx.cpp`:
- `if constexpr` in a **non-template** function (such as the kernel itself)
  still type-checks the discarded branch, so move layout choices into
  templates.
- Inside a template, code that does not depend on a template parameter is
  checked when the template is *defined*, even if it is never
  instantiated. Make it depend on a parameter, for example
  `constexpr int W = ROWS > 0 ? NP : NP;`.

## 12. Worked idioms

**Linear filter (paper, Algorithm 2):** load an 8x32-byte block with one
2D media read. The nine 3x3 neighbours are `select<6,1,24,1>(dy, dx)` of the
same registers, and only one load and one store go to memory. Its 6x24
char-to-float select compiles to 9 SIMD16 `mov`s with regions such as
`r4.3<8;8,1>:ub`. See `examples/linear_genx.cpp`.

**2x2 transpose by replicate + merge (paper):**
```cpp
vector<int, 4> v;                               // [a, b, c, d] = [[a, b], [c, d]]
vector<int, 4> v0 = v.replicate<2, 1, 2, 0>(0); // [a, a, b, b]
vector<int, 4> v1 = v.replicate<2, 1, 2, 0>(2); // [c, c, d, d]
vector<int, 4> t;
t.merge(v0, v1, 0b0101);                        // [a, c, b, d]
```
**8x8 transpose in registers:** `t.row(c) = m.column(c)` for c in 0..7, or
equivalently `t.row(c) = m.format<float>().select<8, 8>(c)`
(`examples/patterns_genx.cpp`, `transpose8`).

**Variable SIMD width:** to save registers in high-pressure phases, run
SIMD8 or SIMD16 slices of a larger vector (`v.select<16,1>(k*16)`) and use
the full width where bandwidth matters. The SIMT model can't do this; CM
lets you choose per instruction.
