# C for Metal memory access reference

Spec sections 4.4-4.7, 4.19, 7 and the cmocl runtime guide. Every function
moves a whole vector or matrix per message. Pick the message type that
matches the access pattern; that choice matters more for performance than
anything else in a CM kernel.

## Contents
1. Surfaces, stateful vs stateless, kernel argument annotations
2. Legacy dataport API (any Gen9+ target, all cmc versions)
3. Atomics
4. Shared local memory (SLM) and barriers
5. SVM / stateless pointers
6. LSC API (cm_load / cm_ptr_load / cm_atomic, DG2+) -- tested
7. Size, alignment and out-of-bounds rules: cheat sheet, array tails (padding)
8. Host-side binding (OpenCL / Level Zero / SYCL)

---

## 1. Surfaces and argument annotations

- `SurfaceIndex` is a binding-table index. Accesses through it are
  **stateful**: 32-bit offsets, hardware bounds checking, at most 4 GB.
- `svmptr_t` (or a `__global T*` with newer cmc) is a raw 64-bit address.
  Accesses through it are **stateless**: no bounds check and no size limit.
- When CM runs on the OpenCL or Level Zero runtime, each kernel argument
  must say what it is:

| `[[type("...")]]` | Host passes | Used with |
|---|---|---|
| `buffer_t` | `cl_mem` buffer, USM pointer (SYCL/L0) | OWord block, scattered, atomics, LSC stateful |
| `svmptr_t` | SVM pointer (`clSetKernelArgSVMPointer`), USM pointer | `cm_svm_*`, `cm_ptr_*` |
| `image2d_t` | `cl_mem` 2D image | **media block** read/write (`examples/linear_genx.cpp`); a plain sampler image only with the backend option `-vc-use-plain-2d-images` |
| `image2d_media_block_t` | 2D image | media block read/write; `image2d_t` is the portable spelling (the 2021 cmc ignores this one, `legacy-sdk.md`) |
| `image1d_t`, `image1d_buffer_t`, `image1d_array_t`, `image2d_array_t`, `image3d_t` | images | typed and sampler access |
| `sampler_t` | sampler | `sample16`, `cm_3d_sample` |
| access modifiers `read_only`, `write_only`, `read_write` | | 2D surfaces only, e.g. `[[type("image2d_t read_only")]]` |

Without an annotation the OpenCL runtime cannot bind the argument, and
the behaviour is undefined. The 2021 cmc also warns that it ignores
`[[type("svmptr_t")]]` on an `svmptr_t` parameter, but the parameter still
works as a 64-bit value set with `clSetKernelArgSVMPointer`.

Surfaces may not alias each other (ordering between typed and untyped
accesses is not guaranteed).

## 2. Legacy dataport API

### OWord block read/write (buffers)
```cpp
void read(SurfaceIndex buf, int byte_offset, vector_ref<T, N> v);   // also matrix
void read(DWALIGNED(buf), int byte_offset, vector_ref<T, N> v);     // offset only 4B aligned
void write(SurfaceIndex buf, int byte_offset, const vector<T, N> v);
```
- Read: offset 16 B aligned (4 B with `DWALIGNED`). Write: offset 16 B
  aligned, size 1, 2, 4 or 8 OWords (16, 32, 64 or 128 bytes).
- **At most 128 bytes per call.** For 256 bytes, read into
  `v.select<32,1>(0)` and `v.select<32,1>(32)`.
- Out-of-bounds reads return 0; out-of-bounds writes are dropped.

### Scattered (gather/scatter, buffers)
```cpp
void read (SurfaceIndex buf, uint global_off, vector<uint, N> elem_off, vector_ref<T, N> v);
void write(SurfaceIndex buf, uint global_off, vector<uint, N> elem_off, vector<T, N> v);
void write(SurfaceIndex buf, uint global_off, uint elem_off, T scalar);   // single element
void read_scaled / write_scaled(...)   // same, but offsets in BYTES and unaligned
```
- `read`/`write` offsets are in **units of the element size**, N is a power
  of 2 up to 32, and T is a 1-, 2- or 4-byte type or float.
- They are allowed inside SIMD control flow when N equals the block width.
- Overlapping scatter addresses: the write order is undefined.

### 2D media block read/write (2D surfaces)
```cpp
void read (SurfaceIndex img, int x_bytes, int y_rows, matrix_ref<T, R, C> m);
void write(SurfaceIndex img, int x_bytes, int y_rows, const matrix<T, R, C> m);
```
- `x` is in **bytes** and `y` in rows. One message loads a whole 2D block
  (rows can be 4-64 bytes wide).
- Limits on width (bytes) and height (rows): 1-4: 64, 5-8: 32, 9-16: 16,
  17-32: 8, 33-64: 4.
- Reads past the edge clamp to the nearest edge pixel; out-of-bounds writes
  are dropped. Write x and width must be multiples of 4 bytes.
- Not allowed inside SIMD control flow.
- Host: a `cl_mem` image (`clCreateImage`, e.g. `CL_R`/`CL_UNSIGNED_INT8`, so
  that a pixel is one byte). The argument is annotated as described above.

### Others (spec 4.6)
Typed surface read/write (`read_typed`, `write_typed`), untyped
`read_untyped`/`write_untyped` with channel masks, planar media block
(`read_plane`), sampler `sample16`/`load16`/`cm_3d_sample`/`cm_3d_load`
(spec 4.8).

## 3. Atomics

```cpp
template <CmAtomicOpType Op, typename T, int N>
void write_atomic([vector<ushort, N> mask,] SurfaceIndex buf, vector<uint, N> dword_off,
                  [vector<T, N> src0, [vector<T, N> src1,]] [vector_ref<T, N> old]);
write_atomic<ATOMIC_INC, uint>(hist, bins);                 // no sources
write_atomic<ATOMIC_ADD, uint>(buf, offs, vals);            // one source
write_atomic<ATOMIC_CMPXCHG, uint>(buf, offs, newv, cmpv, old);
```
- The ops are `ATOMIC_ADD SUB INC DEC MIN MAX XCHG CMPXCHG AND OR XOR MINSINT
  MAXSINT PREDEC REVSUB` and the float ones `FMAX FMIN FCMPWR`. Every op
  returns the old value.
- Offsets are in **DWords**. N can be 1, 2, 4, 8, 16 or 32.
- **Use SIMD16 messages** (tested on an Arc A770 and a UHD 750).

  | Message width | DG2 (A770), no flag | DG2 + `-ftranslate-legacy-memory-intrinsics` | Gen12LP (UHD 750) |
  |---|---|---|---|
  | SIMD16 | correct | correct | correct |
  | SIMD32 | GPU hang / `CL_OUT_OF_RESOURCES` | correct | loses half the increments |

  These results hold for IGC 2.42 and 2.11, and for SPIR-V from the 2021
  and the current cmc. The flag translates legacy messages to LSC. It is
  only for LSC targets (DG2+): Gen12LP fails to build with it. The skill's
  scripts still pass it on DG2 as a safe default.
- `cm_svm_atomic` and `cm_slm_atomic` are the SVM and SLM variants.

## 4. SLM and barriers (spec 4.7)

```cpp
cm_slm_init(bytes);                 // per group, at most 64 KB; call first, IN THE KERNEL BODY
uint slm = cm_slm_alloc(bytes);     // a HANDLE to the buffer (kernel body only, too)
cm_slm_block_write(slm, byte_off, vector<T, N>);        // 16 B aligned; slot offset goes HERE
cm_slm_block_read(slm, GENX_NONE | GENX_DWALIGNED, byte_off, vector_ref<T, N>);
cm_slm_read(slm, vector<ushort, N> elem_addr, vector_ref<T, N>);   // gather
cm_slm_write(slm, vector<ushort, N> elem_addr, vector<T, N>);      // scatter
cm_slm_read4 / cm_slm_write4(slm, vector<uint,N> addr, v, SLM_ABGR_ENABLE ...) // 4 dwords/address, much faster, data transposed
cm_slm_atomic(slm, ATOMIC_ADD, addr, dst, src0);
cm_slm_load(slm, SurfaceIndex mem, mem_off, bytes); // cooperative global->SLM copy (+ barrier), bytes % 256 == 0
cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);  // before the barrier, so writes are visible
cm_barrier();                            // group-wide barrier
cm_sbarrier(1) / cm_sbarrier(0);         // split barrier: signal / wait
```
**Tested pitfalls:**
- **`cm_slm_alloc` returns a handle, not an address.** Put a thread's slot
  offset in the offset argument: `cm_slm_block_write(slm, lid * 16, v)`.
  With the 2021 cmc, `cm_slm_block_write(slm + lid * 16, 0, v)` compiles
  without a warning, but the addition is ignored: all 8 threads of a group
  wrote slot 0, and the read-back gave `2 0 0 0 0 0 0 0` instead of
  `1 2 ... 8`. (The current cmc happens to honour it; don't rely on that.)
- **`cm_slm_init` and `cm_slm_alloc` must be called in the kernel
  function**, not in a helper: "cm_slm_init shall only be called in a
  kernel". Allocate in the kernel and pass the handle down.

Host side: the local size is the number of **threads** per group (8 in
`examples/patterns_genx.cpp` `group_sum`). The SLM size is fixed by the
kernel; the host does not pass it.

## 5. SVM / stateless (spec 4.19)

```cpp
extern "C" _GENX_MAIN_ void k(svmptr_t p [[type("svmptr_t")]]) {
  vector<float, 32> v;
  cm_svm_block_read(p + byte_off, v);          // 16 B aligned
  cm_svm_block_read_unaligned(p + off, v);     // 4 B aligned
  cm_svm_scatter_read(vector<svmptr_t, N> addrs, vector_ref<T, N>);
  cm_svm_block_write(p + byte_off, v);
  cm_svm_scatter_write(addrs, values);
  cm_svm_atomic(ATOMIC_ADD, addrs, old, src0);
}
```
`svmptr_t` is byte-addressed, so `p + 1` is the next byte. Pointer size:
`-DCM_PTRSIZE=64` (the default with `-m64`). Host side: `clSVMAlloc` with
`clSetKernelArgSVMPointer`, or USM pointers through Level Zero or SYCL.
Out-of-bounds accesses are undefined (there is no bounds check).

The headers also provide `cm_ptr_block_read`, `cm_ptr_block_write` and
`cm_ptr_block_read_unaligned` for typed `T*` pointers.

## 6. LSC API (`CM_HAS_LSC`, DG2 and newer) -- tested

`cm_lsc.h` comes with `intel-igc-cm` (tested with 1.0.225); the 2021 SDK lacks it. `examples/lsc_genx.cpp` +
`host_lsc.cpp` exercise it and PASS on DG2 (an A770) from `.spv`, `.bin` and
CM source. Use the "new interface" with an explicit element count; the
`VectorSize` forms of block messages emit a deprecation warning.

```cpp
// block, stateful: ONE address, NElts contiguous U32/U64 ("transposed" message)
vector<float, 64> v = cm_load<float, 64>(buf, byte_off);              // load.ugm.d32x64t
cm_store<float, 64>(buf, byte_off, v);
// cache hints: L1 then L2; invalid pairs fail at compile time
v = cm_load<float, 64, DataSize::Default, CacheHint::Streaming, CacheHint::Cached>(buf, off);   // .st.ca
cm_store<float, 64, DataSize::Default, CacheHint::Uncached, CacheHint::WriteBack>(buf, off, v); // .uc.wb
// block, stateless (svmptr_t / USM / SVM pointer)
v = cm_ptr_load<float, 64>(reinterpret_cast<const float *>(p), byte_off);
cm_ptr_store<float, 64>(reinterpret_cast<float *>(p), byte_off, v);
// gather: per-lane BYTE offsets, SIMD16 on DG2 (SIMD32 from PVC: details::lsc_default_simt())
vector<float, 16> g = cm_load<float>(buf, offsets);                    // load.ugm.d32
// atomics: AtomicOp::IINC/IADD/FADD/SMIN/UMAX/ICAS/..., byte offsets, SIMD16 on DG2
cm_atomic<AtomicOp::IINC, uint>(bins, offsets);                        // atomic_iinc.ugm.d32
// also: cm_prefetch / cm_ptr_prefetch, cm_load4 / cm_store4 (quad), cm_load_slm / cm_store_slm,
// cm_fence(LSC_SFID, LSC_FENCE_OP, LSC_SCOPE)
```

- **Units:** every LSC offset is in **bytes**. That includes gathers and
  atomics, where the legacy API used elements and DWords.
- **Block sizes:** NElts is one of {1, 2, 3, 4, 8, 16, 32, 64} U32/U64
  elements, so up to 256 B per message against the legacy OWord block's
  128 B. Offsets must be aligned to the data size.
- **Out of bounds:** stateful out-of-bounds reads return 0; stateless ones
  are undefined.
- **Not on DG2:** `CM_HAS_LSC_CONTROL` makes every LSC call a compile error
  on pre-LSC targets (`rkl`, `tgllp`). Within LSC targets:
  - 2D block messages (`lsc/block2d.h`, `block_2d_desc`) need
    `CM_GENX >= 1280` (PVC, Xe2: BMG, LNL);
  - typed LSC (images) needs Xe2 (`__CM_INTEL_TARGET_MAJOR >= 20`).
  - So on DG2, 2D tiles stay on the legacy media block (`read(image,
    x, y, m)`, `examples/linear_genx.cpp`).
- **Performance (measured on an A770, 256 MiB copy):** LSC 256 B blocks did **not** raise
  bandwidth over legacy 128 B OWord blocks. Both reach 420-450 GB/s at best,
  and single runs vary from 150 to 450 GB/s. The backend already lowers
  legacy OWord reads on DG2 to `load.ugm.d32x32t` (with
  `-ftranslate-legacy-memory-intrinsics`). What LSC adds is half the send
  messages per byte, cache hints, stateless block access without
  `cm_svm_*`, float and min/max atomics, and the 2D blocks of newer GPUs.
  Check the generated messages in the IGC asm (`cm_build.sh -k`, then
  `grep -oE '(load|store|atomic_[a-z]+)\.ugm[^ ]*' *.asm`).

## 7. Cheat sheet

| Message | Address unit | Align | Max per call | OOB read | OOB write |
|---|---|---|---|---|---|
| OWord block `read/write(buf, off, v)` | bytes | 16 B (4 B with DWALIGNED, read only) | 128 B | 0 | dropped |
| scattered `read/write(buf, g, offs, v)` | element | element | 32 lanes | 0 | dropped |
| `read_scaled/write_scaled` | bytes | none | 32 lanes | 0 | dropped |
| media block `read/write(img, x, y, m)` | x bytes, y rows | x pixel (read), 4 B (write) | 64 B x 4 ... 4 B x 64 | edge clamp | dropped |
| `write_atomic` | DWords | 4 B | 32 lanes (use 16) | 0 | dropped |
| `cm_svm_block_read/write` | bytes | 16 B | 128 B | undefined | undefined |
| `cm_slm_block_read/write` | bytes | 16 B (4 B DWALIGNED read) | 128 B | 0 | dropped |
| LSC `cm_load/cm_store` (block) | bytes | data size | 64 elements (256 B) | 0 | dropped |
| LSC gather / `cm_atomic` | bytes | data size | 16 lanes (DG2) | 0 | dropped |

### Array tails: pad the arrays, or store the tail exactly

A block message always moves the whole block. If the array length n is not
a multiple of what one thread moves (CHUNK elements, or the block size),
the last thread reads and **writes past element n**.
- **"OOB write: dropped" above means past the end of the surface**, i.e.
  the `cl_mem` allocation, not past n. Elements between n and the end of the
  allocation are written: a padded tail, or another array in the same
  allocation (sub-buffers, one pool with offsets).
- **USM/SVM has no check at all.** The overrun lands in the next
  allocation or the allocator's metadata (heap corruption later), or
  faults.
- **Reads** past n return 0 only past the end of a `cl_mem`. Otherwise
  they return whatever is there, which then flows into sums and maxima.
- OWord blocks move multiples of 16 B, so an array whose length isn't a
  multiple of 4 floats can't be written exactly with blocks at all.

Fixes, in order of preference:
1. **Pad:** allocate `round_up(n, CHUNK)` elements, or whole thread-group
   multiples when the grid is rounded up. Zero the pad on input arrays,
   and ignore it on output arrays. On the host: `cmsycl::to_device(q, v,
   pad)`, `host_spmv.cpp`, `host_gesv.cpp` (lanes mode).
2. **Store the tail exactly:** the last thread switches to scattered
   element writes for its remaining elements, as `gesv_genx.cpp` does for
   odd N.
3. **Rely on surface bounds only when the array is the whole `cl_mem`**,
   as the SDK's `BufferTest` does. That breaks the moment the host moves to
   USM (SYCL) or packs arrays together.

## 8. Host-side binding

- **SYCL, the skill's default** (`examples/cm_sycl.hpp`): every surface kind
  except images maps to a USM pointer.

  | Kernel parameter | Host object |
  |---|---|
  | `SurfaceIndex [[type("buffer_t")]]` | USM pointer (`cmsycl::to_device(q, v, pad)`, `sycl::malloc_device/shared`) |
  | `svmptr_t` | the same USM pointer |
  | scalars | plain values |
  | `image2d_t` | not reachable through kernel_compiler: use OpenCL |

  Launch with `cmsycl::launch(q, k, threads, per_group, args...)` or
  `launch2d(q, k, tx, ty, gx, gy, args...)` in CM order. SYCL reverses
  dimensions, its last range dimension being the backend's dimension 0.
  **USM is not bounds-checked**, unlike `buffer_t` on a `cl_mem`, where
  out-of-range reads return 0 and writes are dropped. Pad allocations
  that a kernel's last block or thread overruns. `make_queue` is
  in-order, because USM carries no dependencies.
- **OpenCL, using the C++ bindings** (CLHPP, `<CL/opencl.hpp>`, as in
  `examples/cm_ocl.hpp`), when the user picks it, and for images and spec constants:

  | Kernel parameter | Host object |
  |---|---|
  | `SurfaceIndex [[type("buffer_t")]]` | `cl::Buffer` (`cl::Buffer(ctx, v.begin(), v.end(), readOnly)` copies a vector) |
  | `SurfaceIndex [[type("image2d_t")]]` (media block) | `cl::Image2D(ctx, flags, cl::ImageFormat(CL_R, CL_UNSIGNED_INT8), w, h, 0, ptr)` |
  | `svmptr_t` | raw pointer from `cl::SVMAllocator`; a `T *` functor argument goes through `clSetKernelArgSVMPointer` |
  | scalars | plain values (`cl_uint`, `float`) |

  Launch with
  `cl::KernelFunctor<Args...> k(prog, "name"); k(cl::EnqueueArgs(q, cl::NDRange(threads...), cl::NDRange(per_group...)), args...).wait();`,
  where the NDRange counts **threads**. Read results with
  `cl::copy(q, buf, v.begin(), v.end())` or `q.enqueueReadImage(...)`.
  Global work offset is not supported.
  Coarse-grained SVM through CLHPP (`cl::SVMAllocator<T, cl::SVMTraitCoarse<>>`)
  returns memory that is **already mapped**: fill it, `cl::unmapSVM(v)` every
  buffer before the launch, `cl::mapSVM(v)` to read results. Calling
  `mapSVM` again before writing left a buffer mapped during the kernel and
  every result was wrong. Both use the default queue, so call
  `cl::CommandQueue::setDefault(q)` first.
- **Level Zero:** `zeKernelSetArgumentValue(k, i, sizeof(void*), &usm_ptr)`
  for buffers. `zeKernelSetGroupSize(k, lx, ly, lz)` sets threads per group
  and `zeCommandListAppendLaunchKernel` takes the group count.
