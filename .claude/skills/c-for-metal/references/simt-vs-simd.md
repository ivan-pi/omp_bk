# SIMT vs. explicit SIMD, and where sub-groups fit

## Contents
1. The three models side by side
2. Sub-groups (SYCL/OpenCL) vs. ESIMD/CM
3. The same job three ways (tested)
4. Choosing
5. cuSolverDx-style "Thread" and "Block" operators in CM

---

## 1. The three models side by side

| Concept | CPU explicit SIMD (AVX-512, GNU vectors) | CM / ESIMD (explicit SIMD on the GPU) | CUDA / OpenCL / SYCL SPMD (implicit, SIMT) |
|---|---|---|---|
| unit of code | one CPU thread | one GPU hardware thread | one lane (work-item) |
| vector width | in the type (`__m512`, `vector_size`) | in the type (`vector<T,N>`/`simd<T,N>`, any N; the compiler splits it into SIMD8/16/32) | chosen by the compiler (SIMD8/16/32 per kernel) = the sub-group size |
| registers | compiler allocates from your vectors | your vectors *are* register regions (GRF), sized by you | compiler allocates per lane |
| cross-lane data | permute/shuffle intrinsics | `select`/`replicate`/`iselect`/`format`: free register regioning | sub-group shuffles and collectives, SLM |
| divergence | masks / blends | `merge`, masks, or `SIMD_IF` (per lane) vs. scalar `if` (uniform) | implicit, handled by the compiler |
| uniform values | ordinary scalars | ordinary scalars | the compiler must prove a value uniform to keep it scalar |
| memory | vector loads/gathers | block / 2D media / scattered / atomic messages, chosen by you | per-lane loads, which the compiler coalesces; sub-group block reads |

## 2. Sub-groups (SYCL/OpenCL) vs. ESIMD/CM

The confusing part is that **both describe the same hardware object**.
- An Intel GPU hardware thread runs SIMD instructions 8, 16 or 32 lanes
  wide.
- A SYCL/OpenCL kernel is compiled at one SIMD width. The compiler packs that
  many consecutive work-items into one hardware thread, and that bundle is
  the **sub-group**.
- `sub_group::get_max_local_range()` reports the width, and
  `[[sycl::reqd_sub_group_size(16)]]` (`intel_reqd_sub_group_size` in OpenCL
  C) requests one.
- An **ESIMD or CM work-item is the whole hardware thread**, which is why their
  NDRange counts threads, the "SIMD1 dispatch".

So the work done by a sub-group of 16 work-items, each holding one `float v`,
is the same as one CM/ESIMD thread holding `vector<float, 16> v`. Lane *i*
of the sub-group is element *i* of the vector, and the data even sits in the
same registers.

The differences are about **what you can express and who decides**:

| | Sub-group (SIMT view) | ESIMD / CM (explicit view) |
|---|---|---|
| what your code sees | its own lane's values; other lanes only through group functions | the whole register file; any element of any vector |
| cross-lane exchange | `select_from_group`, `shift_group_left/right`, `permute_group_by_xor`, `group_broadcast`; reductions and scans with `reduce_over_group`, `inclusive/exclusive_scan_over_group`, `any_of_group`/`all_of_group` | direct register regions: `select<N,S>(i)`, `replicate`, `iselect`, `format`; `cm_sum`/`reduce`; any/all on masks |
| data shape | exactly one value (or a few) per lane: width = sub-group size | any shape: `vector<float,144>`, `matrix<uchar,8,32>`; one statement can be wider or narrower than the SIMD width |
| SIMD width | fixed for the whole kernel (8/16/32) | chosen per statement (`select<8,1>` → SIMD8, a 32-wide add → SIMD32) |
| divergence | implicit per-lane `if` (the compiler masks) | explicit `merge`/`SIMD_IF`; scalar `if` is uniform |
| register control | none: the compiler decides spills and layout | you size the working set, which must fit the GRF |
| memory | per-lane loads coalesced by the compiler; `sycl::ext::oneapi::experimental::group_load` or Intel sub-group block reads for one wide message | one block, 2D media or LSC message per vector or tile, by choice |
| matrix engines | `joint_matrix` (cooperative over the sub-group) | `xmx::dpas` / `cm_dpas` on register operands you lay out (VNNI) |
| bridge | `invoke_simd(sg, f, args...)` calls an ESIMD function `f` with the sub-group's per-lane values packed into `simd<T, sg_size>`; `uniform<T>` arguments stay scalar | ESIMD functions called from SPMD code receive exactly that sub-group vector |

Cost model: a sub-group shuffle with a **constant** offset compiles to the
same register region move that CM writes with `select`. A **variable**-index
shuffle becomes indirect register addressing, and its cost is the same in
both models. Group reductions and scans compile to the log-step sequences
that CM writes explicitly. So the hardware operations are identical. What
ESIMD/CM adds is: data that is not lane-shaped (tiles, rows wider than the
sub-group), per-statement SIMD width, explicit register budgeting, and the
choice of memory message.

## 3. The same job three ways (tested)

Each group of 16 consecutive floats → its sum and its inclusive prefix
scan. `examples/esimd/subgroup_vs_esimd.cpp` runs (1) and (2); on an A770
they agreed bit for bit. (3) is the CM form used in
`examples/scan_genx.cpp`.

```cpp
// (1) SYCL sub-group: one work-item per element; 16 of them form the sub-group.
q.parallel_for(nd_range<1>{n, 128}, [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
  sub_group sg = it.get_sub_group();
  float v = x[it.get_global_id(0)];
  scan[it.get_global_id(0)] = inclusive_scan_over_group(sg, v, plus<float>());
  float s = reduce_over_group(sg, v, plus<float>());
  if (sg.get_local_id()[0] == 0) sum[it.get_global_id(0) / 16] = s;
});

// (2) ESIMD: one work-item = the hardware thread, owning all 16 values.
q.parallel_for(range<1>{n / 16}, [=](id<1> t) SYCL_ESIMD_KERNEL {
  simd<float, 16> v = block_load<float, 16>(x + t * 16), s = v;
  { simd<float, 16> sh = 0; sh.select<15, 1>(1) = s.select<15, 1>(0); s += sh; } // and 2, 4, 8
  ...
  block_store(scan + t * 16, s);
  sum[t] = reduce<float>(v, std::plus<>());
});

// (3) CM: the same as (2), written with vector<float,16>, select<> and cm_sum<float>,
//     launched with NDRange = n / 16 threads (examples/scan_genx.cpp, scan16_demo).
```

## 4. Choosing

- **SYCL with sub-groups** when the data is naturally one element per lane
  and cross-lane traffic is occasional (a reduction, a broadcast, a
  shuffle). It is portable SYCL, and current Intel compilers do this well.
- **ESIMD** (current toolchains) or **CM** (legacy) when the per-thread data
  is a tile or a row wider than the SIMD width, when a computation mixes
  widths, or when you need control of registers, memory messages or
  `dpas` operand layout. They also let you combine both models:
  `invoke_simd` puts an ESIMD inner kernel inside a SYCL SPMD kernel.

## 5. cuSolverDx-style "Thread" and "Block" operators in CM

cuSolverDx (and cuBLASDx) offer **Thread** execution, where one CUDA thread
solves a whole small problem in its own registers and a warp holds 32
independent problems, and **Block** execution, where a thread block
cooperates on one problem through shared memory and `__syncthreads()`. In
CM, a "thread" is the whole SIMD hardware thread, so the mapping shifts by
one level:

| cuSolverDx | Who works on one problem | CM equivalent | Status in this skill |
|---|---|---|---|
| **Thread** operator | one SIMT lane; a warp holds 32 problems | **SIMD across the batch**: one CM thread holds 16/32 problems, lane *i* = problem *i* (structure-of-arrays: element (r,c) of all problems in one `vector<float,16>`). Pivot rows differ per lane, so a row swap becomes per-lane `merge`s over the candidate rows instead of one register move | implemented: `examples/gesvb_genx.cpp`, N = 2..8 |
| a **warp-level** operator, or a **Block** operator small enough for one warp | 32 lanes cooperating | **one CM thread per problem, SIMD across the matrix**: the whole system in registers, with cooperation through register regions, no shared memory and no barriers. This is `examples/gesv_genx.cpp` (N <= 16 in 128 GRFs; 24-32 with `-double-GRF`) | implemented, N = 3..32 |
| **Block** operator for larger N | many warps + shared memory + `__syncthreads()` | a **CM thread group**: several hardware threads, each owning a panel or row block, exchanging pivot rows and panels through SLM (`cm_slm_*`) with `cm_barrier()` | not implemented |

In short, what cuSolverDx calls a Block operator, for the sizes that fit
one register file, is what CM does as a single "thread". The SIMT Thread operator corresponds in CM to batch-in-lanes
SoA code. That layout suits very small N (2-8), where one problem would
use only a few SIMD lanes.

Measured on an Arc A770 (1M systems, 1 right-hand side, ms):

| N | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| one system per thread (`gesv_genx.cpp`) | 0.373 | 0.549 | 0.648 | 1.018 | 1.291 | 1.538 | 1.375 |
| batch in lanes, gathers | 0.149 | 0.422 | 0.837 | 1.202 | 1.682 | 2.123 | 2.782 |
| batch in lanes, block reads + register transpose | **0.099** | **0.224** | **0.381** | **0.606** | **0.825** | **1.176** | 1.431 |

The lanes layout wins at small N, where a single small problem fills only a
few SIMD lanes. One system per thread catches up around N = 8 and wins
beyond. How the data is loaded matters as much as the mapping:
per-element gathers (one cache line per lane per message) cost the lanes
version half its bandwidth until it read its contiguous systems with block
messages.

Both are memory-bound as stand-alone kernels. Their natural use is inside
a longer fused kernel ("mega-kernel") that builds, solves and consumes the
small systems in registers, which is the kernel-fusion pattern of
`examples/esimd/pipeline_functor.cpp`.
