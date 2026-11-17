# SYCL kernels

| File | Contents |
|---|---|
| `bk1_sycl.cpp` | BK1 variants, one order per binary (`bk1_sycl_<p>`, AOT; `bk1_sycl_<p>_jit`) |
| `bk1_sycl_sc.cpp` | BK1 opt kernel, every order and shape in one JIT binary (`bk1_sycl_sc`) |
| `bk1_sycl_opt.h` | the BK1 opt kernel, its shape struct `opt_params` and shape table `opt_variants` |
| `bk1_sycl_harness.h` | host side of both BK1 drivers: data, reference, timing, result lines |
| `bk3_sycl.cpp`, `bk5_sycl.cpp` | BK3 and BK5 variants, one order per binary |
| `PERFORMANCE.md` | do's and don'ts for fast SYCL kernels, from what we measured here |
| `repro_simd/` | reproducer for Intel: IGC's SIMD width for BK5's `sf9c_b1x1` kernel, and how a wrapper functor changes it |

Shared: `include/bk_sycl.h` (queue with `BK_PROFILE`/`BK_INORDER`, timing,
`build_specialized`), `include/bk_harness.h` (references and checks).

## Specialization constants

A kernel whose shape (order, batch, work-group size, layouts) comes from a
specialization constant runs every order from one JIT-compiled binary, and its
shape can be tuned at run time (`BK_SHAPE`) without rebuilding. The JIT folds
the constant, so the kernel is as fast as its ahead-of-time twins; ahead of
time the constant is a run-time value and the same kernel is 2.5-5x slower
(BK1, A770: `spec_b1` in `bk1_sycl_4` against `bk1_sycl_4_jit`).

### Conventions for a kernel

1. **One shape struct.** Everything the kernel needs at compile time goes in
   one trivially copyable struct (`bk1opt::opt_params`). The body takes it
   by value from `Shape::get(kh)`: `fixed_shape<...>` returns a `constexpr`
   value (the AOT variants), `sc_shape` reads the `specialization_id`. One
   body serves both.
2. **Launch the kernel functor itself.** Its pointers are its own members,
   and a required sub-group size goes in its `get(syclex::properties_tag)`.
   A wrapper functor (holding the kernel as a member) changes IGC's code
   generation: BK1 opt ran 20% slower in one. `[[intel::kernel_args_restrict]]`
   has no measurable effect on these kernels; see `include/bk_sycl.h` and
   `repro_simd/` for the details.
3. **Build before timing.** The driver builds each selected shape with
   `bk::build_specialized<Kernel, SpecId>(q, value)` and launches it with
   `cgh.use_kernel_bundle`. The build time is then measured on its own,
   and the timed launches carry no compilation.
4. **Output.** The result lines are those of the AOT driver; after each, a
   line `<name> shape <shape> aot <AOT twin or -> build <s> s first <s> s`
   (`aot`: the AOT variant with the same shape, for the comparison). The
   command line is `<sc> <p> [nelmt] [ntests]`, the AOT binaries
   `<aot>_<p> [nelmt] [ntests]`.
5. **Build.** `bk_gpu_target(<sc> sycl_jit)` plus
   `-fsycl-device-code-split=per_kernel`, so a shape builds only its kernel.
6. **Record it.** `scripts/study/spec_compare.sh <aot> <sc>
   results/<machine>/spec_<kernel>_<machine>.csv`, then a row in the table
   below.

### Status

Device GDoF/s of each shape against its AOT twin (median over the shapes of
an order, and the range over all orders); JIT build time per shape with cold
caches (`NEO_CACHE_PERSISTENT=0`, new `SYCL_CACHE_DIR`), median over the
orders' medians; warm: the SYCL persistent cache.

| Kernel | Driver | Shapes per order | sc / AOT, median by order | range | Cold build per shape | Warm | Data |
|---|---|---|---|---|---|---|---|
| BK1 opt | `bk1_sycl_sc` | 22 (`opt_variants`) | 0.97-0.99 (p = 1: 0.97, p = 2-8: 0.98-0.99) | 0.88-1.02 | 0.06 s (p = 1) to 0.14 s (p = 8) | < 1 ms | `results/a770/spec_bk1_a770.csv` |
| BK1 elem | - | - | not converted: private arrays sized by nq (needs capacity tiers) | | | | |
| BK3 cu, sf | - | - | not converted | | | | |
| BK5 sf | - | - | not converted (same structure as BK1 opt) | | | | |

A770, icpx 2026.1, compute runtime 26.22, 2026-11-17; `DOFS=2e7`.

### Findings (BK1 opt, A770)

- **Caches.** `scripts/run.sh` keeps JIT-compiled kernels in
  `$BUILD_DIR/sycl_cache` (`SYCL_CACHE_PERSISTENT=1`). The compute runtime
  also caches every compilation in `~/.cache/neo_compiler_cache`, on by
  default, so a "cold" SYCL cache is usually still warm there; measure cold
  builds with `NEO_CACHE_PERSISTENT=0`. Cold: 1.2-2.7 s for 22 shapes;
  warm: about 0.01 s.
- **Prebuilt bundle against setting the constant per launch:** the same
  device times, host times no lower (median 0.97x at p = 4). The gain is
  that compilation leaves the timed runs.

Experiments whose code was removed again (they are in the git history of
this branch, before the cleanup that followed them):

- **AOT image of one shape** (`-fsycl-add-default-spec-consts-image` with an
  AOT target): the default shape needs no JIT (0.001 s instead of 0.19 s at
  p = 6), but only through a prebuilt bundle (`get_kernel_bundle<executable>`);
  with the constant set per launch the runtime JIT-compiles even the default
  value. Worth it only for one dominant shape.
- **SLM layouts:** the boxes as nq^3 cubes against dense: 0.96-1.03x. Padded
  (`p1`, still a shape option) is worse at p = 8 (18-23 against 26-28
  GDoF/s) and at p = 3, batch 4 (20.4 against 24.6).
- **JxW read in its natural layout** (e, p, q, r) instead of transposed
  (e, r, q, p): 0.78-1.03x, median 0.92, worst at p = 3-5 (best shape -9 to
  -15%). The seven-step kernels keep the transposed copy; the other
  approaches use the natural layout.
