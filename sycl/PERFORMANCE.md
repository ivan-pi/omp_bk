# Writing fast SYCL kernels: do's and don'ts

These are lessons from tuning the BK1, BK3 and BK5 kernels in this directory
on an Intel Arc A770 (DG2), with icpx 2026.1. Each rule says what we
measured. These kernels are memory-bound, at 71–97% of the STREAM bandwidth.
So most gains come from how the compiler lays out loads, stores and
registers, not from the arithmetic.

## Do

### Launch the kernel functor itself
Hold the kernel's pointers as members of the functor you pass to
`parallel_for`. Put a required sub-group or work-group size in the
functor's own `get(syclex::properties_tag)` (`bk::launch_properties` in
`include/bk_sycl.h`), not in a wrapper. Declaring the geometry once, as a
`bk::kernel_shape` the launcher reads (`bk::launch`), keeps the declared
sizes and the actual launch from drifting apart.

- **Why:** a wrapper functor that holds the kernel as a member and calls it
  changes IGC's code generation. For example, it switches global loads from
  32-bit to 64-bit addressing and picks another SIMD width.
- **Measured:** removing the wrappers made BK1's seven-step kernels 17–40%
  faster and BK3 8–15%. The BK1 opt kernel had lost 20% in a wrapper. BK5's
  wrapped launches were 25% slower at some orders and, by luck, 40% faster
  at others (`repro_simd/`).
- **Range dimension:** make it a template parameter of the kernel (BK5's
  `sf_kernel<..., dims>`) rather than a wrapper.

### Try the SIMD width yourself
IGC's automatic choice can be well off. Keep variants with a required
sub-group size of 8, 16 and 32, and measure them.

- **BK5 G9 at nq = 9:** IGC picks SIMD16; SIMD8 is 1.52× faster.
- **At nq = 3:** IGC's own SIMD32 code is 1.22× slower than the SIMD32 code
  it produces when a sub-group size of 32 is requested
  (`repro_simd/README.md`).
- **In the harnesses:** BK5's `_sg8` variants are the fastest at p = 3–6
  and 8, by 1–8%. BK3's fastest kernels are SIMD8 too.

### Make the shape a compile-time constant
Loop bounds, register-array sizes and local-memory strides must be
constants for the loops to unroll and the arrays to stay in registers.
- **Templates:** use them with one binary per order, ahead of time.
- **Specialization constants:** use them with a JIT build. The opt kernel
  then runs at 0.97–0.99× its AOT twins, for every order and shape from one
  binary (`README.md`, "Specialization constants").
- **One struct:** a single struct-valued specialization constant can carry
  the whole shape, as `bk1opt::opt_params` does.

### Compile specialization-constant kernels before timing them
Build each value with `sycl::build` on an input bundle
(`bk::build_specialized`), and launch with `cgh.use_kernel_bundle`.
- **Why:** compilation then stays out of the timed runs.
- **Cost:** a cold JIT takes 0.06–0.24 s per shape. With the persistent
  cache (`SYCL_CACHE_PERSISTENT=1`, set by `scripts/run.sh`) it's under
  1 ms.
- **Device-code split:** use `-fsycl-device-code-split=per_kernel`, so a
  new value compiles only its own kernel.

### Lay data out so neighbouring work-items read neighbouring addresses
In BK1's step 4, the work-item index p varies fastest. Storing JxW
transposed per element, as (e, r, q, p), makes those reads contiguous.
Reading the natural (e, p, q, r) layout instead costs 0–22% (median 8%,
worst at p = 3–5).

### Move contiguous blocks with sub-group block messages
`group_load` / `group_store` with `data_placement_striped`,
`contiguous_memory` and `full_group` move a contiguous span with one
address per message. They make the `inlB*` variants the fastest at p = 5–7.

- **Requirements:** a required sub-group size of 16 and work-groups that
  are whole sub-groups; copy any remainder element by element.
- **Floats:** move them as `uint32_t` and `bit_cast` them. In icpx 2026.1,
  `group_load` of `float` converts the loaded bits as integer values
  (1.0f becomes 1.065e9).

### Write each step's body out in the kernel
Index the local-memory arrays in the kernel itself, not in helper functions
that take the arrays as plain pointers.
- **Why:** through such helpers the arrays lose their local address space,
  and indexing becomes 64-bit.
- **Measured:** writing the steps out took BK1 from about 320 to about
  420 GB/s (`openmp/offload/bk1_offload/grid_inline.h`).

### Measure with the production build's flags
Compiler flags change the outcome:
- **Under `-fp-model=precise`** (this repository's default), IGC's SIMD16
  code for BK5 G9 at nq = 9 is 1.5× slower than SIMD8. Without it, the two
  are within 4%.
- **The fast model** gave nothing for these memory-bound kernels: the best
  variants were within ±2%, and every check still passed.

### Rank kernels by device time, and run benchmarks alone
- **Device time:** use event profiling (`enable_profiling`), excluding the
  warm-up launch, as the best of several runs. Host time adds launch
  overhead of 1–5 GDoF/s at 2·10⁷ DoFs. That overhead varies between runs:
  BK5 OpenMP host times varied by up to 17%.
- **Separate checking from benchmarking:** benchmarks run without the
  double-precision reference (`BK_CHECK=0`), and `scripts/check_gpu.sh`
  checks every variant on its own.
- **One run at a time:** no builds or other GPU work in parallel, and
  modest host memory (a 4 GB cap, 4 threads).

### Check the generated code before explaining a speed-up
Dump IGC's assembly (`IGC_ShaderDumpEnable=1`,
`IGC_DumpToCustomDir=<dir>`). Compare the SIMD width (`_simd<N>_` in the
file name), the instruction count and the load and store messages
(`load.ugm...a32` against `a64`).

We first attributed the wrapper effect to `kernel_args_restrict` "not
reaching" the pointers. The dumps showed identical code with and without
the attribute.

## Don't

### Don't rely on specialization constants ahead of time
An AOT build cannot fold the constant, so it becomes a value read at run
time. The same opt kernel then runs 2.5–5× slower than with a JIT build.

### Don't expect `[[intel::kernel_args_restrict]]` to help
On these kernels it made no measurable difference: the code is identical
with or without it (BK1 opt within ±1%; BK5 in `repro_simd/`). It's harmless
to keep, but it's not a tuning knob.

### Don't launch one functor type over two range shapes
A 1D and a 2D launch of the same functor type give two kernels with the same
name, and the compile fails ("definition with same mangled name"). Make the
dimension part of the type.

### Don't assume padding the local memory helps
- **Padded strides against bank conflicts:** 25% slower at p = 8 (18–23
  against 26–28 GDoF/s) and 17% slower at p = 3 with batch 4. They help
  elsewhere, so keep them as variants and measure.
- **nq³ boxes for every step:** 0.96–1.03× of the dense layout, no gain.

### Don't trust IGC's choice for register-heavy kernels
Kernels that keep several arrays of nq floats per work-item, like BK5's G9
kernel with 3 × nq + 9 values, are where the automatic SIMD width goes wrong
(see "Try the SIMD width yourself"). Always measure explicit sub-group sizes
for them.

### Don't benchmark from a single host-timed run
Variants within a few percent of each other swap places between runs. Pick
by device time, take the median of several runs (`scripts/study/bk1_pick.sh`
uses 3), and repeat any surprising result before acting on it.
