# Drafts for intel/tiny-tensor-compiler (not posted)

Environment for all three: Arc A770 (acm-g10), tinytc v0.5.0 (ac6cbc7), oneAPI 2026.1,
Level Zero driver 1.15.38646+6, Ubuntu 24.04.

---

## 1. Feature request: Xe-HPG / DG2 (Arc A-series) as a target

tinytc runs on an Arc A770 through the JIT path (`create_core_info(device)`), and our CEED
bake-off kernels (BK1, BK3, BK5 in f32, cooperative matrices with sub-groups 8 and 16) pass and
match or beat hand-written SYCL there. Two gaps:

- There is no offline target: `tinytc -d acm-g10`, `-d dg2` and `-d xe_hpg` fail with
  `Error (2): Invalid arguments`. We check DG2 kernels with `-d tgl`, which has the same
  sub-group sizes (8/16/32) and register size, but that is a guess, not a target.
- `get_support_level` returns `basic` for DG2, since only PVC and BMG are `tuned`
  (`src/ze/device_info.cpp`), and `core_info_intel` sets no matrix-extension/block-IO table
  for it (`src/device_info.cpp`).

Would you accept a `dg2` architecture (IP version of acm-g10/g11/g12) for
`intel_gpu_architecture`, the `-d` option and `create_core_info_intel_from_name`? And is
DPAS (XMX, sub-group 8 on Xe-HPG) for f16/bf16/i8 cooperative matrices planned for it? We'd be
glad to test on an A770.

`tinytc-ze-info` on the A770: sub-groups 8 16 32, register space 4096, float16 yes,
float64 no, bfloat16_conversion no, subgroup_buffer_block_io yes.

---

## 2. Bug: local memory over the device limit fails with an undiagnosed runtime error

A kernel whose `alloca ... local` exceeds the per-work-group local memory (65536 bytes on the
A770) fails at kernel creation with only

```
Source name unavailable:0.0:
tinytc error 18: Error occured in compute runtime
```

Underneath, `zeModuleCreate` returns `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` with an empty
build log, which points at device memory (16 GB free) rather than local memory. tinytc knows
each alloca's size and stack offset after `set-stack-ptr`; a check against the device's local
memory size (or at least a message naming local memory) would save a lot of searching.

Minimal reproducer: a single `alloca : memref<f32x$N,local>`; N = 16384 runs, N = 16385
fails. Reproducer: `tinytc/repro_slm/` (repro.cpp, build.sh, run.sh, outputs, Level Zero trace).

(Also: "occured" → "occurred" in the message.)

---

## 3. Question/bug: `insert-barrier` ignores local memory used inside `foreach_tile`

The manual (tensor-ir.rst) says barriers are inserted automatically in collective regions
and manual barriers should only be needed in SPMD regions. Two consecutive `foreach_tile`
loops at function (collective) level, the first storing tiles to an `alloca` buffer and the
second loading them, get no barrier:

```
func @fb(%A: memref<f32x64x8>, %B: memref<f32x8x8>, %C: memref<f32x64x8>)
attributes{subgroup_size=16,work_group_size=[64,1]} {
    %c0 = constant 0 : index
    %c64 = constant 64 : index
    %w = alloca : memref<f32x64x8,local>
    foreach_tile (%t)=(%c0),(%c64) as (%n)<=(16) {
        %a = cooperative_matrix_load %A[%t,%c0] : coopmatrix<f32x16x8,matrix_acc>
        cooperative_matrix_store %a, %w[%t,%c0]
    }
    foreach_tile (%t)=(%c0),(%c64) as (%n)<=(16) {
        %x = sub %c64, %t : index
        %r = sub %x, %n : index          ; read rows another sub-group wrote
        %b = cooperative_matrix_load %w[%r,%c0] : coopmatrix<f32x16x8,matrix_a>
        %d = cooperative_matrix_load %B[%c0,%c0] : coopmatrix<f32x8x8,matrix_b>
        %z = constant 0.0 : coopmatrix<f32x16x8,matrix_acc>
        %e = cooperative_matrix_mul_add %b, %d, %z : coopmatrix<f32x16x8,matrix_acc>
        cooperative_matrix_store %e, %C[%t,%c0]
    }
}
```

`tinytc-opt -d tgl -pinsert-barrier fb.ir` prints the kernel unchanged: no `barrier.local`
between the two loops.

In `insert_barrier_pass::run_on_region`, the loads and stores inside the loop are in an SPMD
region (`kind_max >= spmd`, so skipped), and the `foreach_tile` instruction itself contributes
no reads or writes (`get_rw` falls through to `inst_view`). Summarizing a foreach body's memory
effects onto the foreach instruction would make the collective-level rule hold. Until then, a
sentence in the manual that `foreach`/`foreach_tile` need manual barriers between them would
help: missing ones give wrong results silently.
