# IGC picks a slow SIMD width for a register-heavy SYCL kernel on DG2

Built with `-fp-model=precise`, IGC's automatic SIMD width for this kernel runs
1.2–1.5× slower on an Arc A770 than a width we request ourselves:

- **nq = 9:** IGC compiles the kernel for SIMD16. A required sub-group size of
  8 (SIMD8) runs 1.52× faster.
- **nq = 3:** IGC compiles it for SIMD32. The SIMD32 code it produces when we
  require a sub-group size of 32 runs 1.22× faster than its own SIMD32 code.
  SIMD8 and SIMD16 are faster than IGC's choice too.

An unrelated change to the source flips IGC's choice. Wrapping the kernel
functor in another functor, which holds it as a member and calls it, gives
SIMD8 at nq = 9 and SIMD16 at nq = 3. The wrapper also switches the global
loads from 32-bit to 64-bit addressing (`load.ugm...a32` becomes `a64`). The
wrapped kernel is faster at both orders, though it does more work.

`[[intel::kernel_args_restrict]]` makes no difference: the kernels with and
without it compile to the same code.

We found this through a variant of one of our benchmark kernels that looked
1.4× faster without the attribute. It turned out to be the wrapper and the
SIMD width it led to, not the attribute.

## Results

`./repro_<nq>` built with `-fp-model=precise` (the default of `build.sh`, as in
our benchmarks); device time, best of 20 launches, 2·10⁷ DoFs, A770:

| Case | nq = 3: SIMD | GDoF/s | nq = 9: SIMD | GDoF/s |
|---|---|---|---|---|
| 1D range, `kernel_args_restrict` | 32 (IGC) | 7.51 | 16 (IGC) | 5.65 |
| 1D range | 32 (IGC) | 7.51 | 16 (IGC) | 5.66 |
| 2D range | 32 (IGC) | 7.51 | 16 (IGC) | 5.68 |
| 2D range, in a wrapper functor | 16 (IGC) | 8.76 | 8 (IGC) | 8.02 |
| required sub-group size 8 | 8 | 8.40 | 8 | **8.58** |
| required sub-group size 16 | 16 | 8.31 | 16 | 5.66 |
| required sub-group size 32 | 32 | **9.13** | 32 | 5.69 |

The SIMD widths are those of IGC's assembly (`asm_<nq>/*_simd<N>_entry_*.asm`,
dumped by `build.sh`). All cases give identical results, and the program exits
with 1 if any differs.

Instruction counts from the dumps:

- **nq = 3:** IGC's SIMD32 code has 964 instructions; the code for a required
  sub-group size of 32 has 925.
- **nq = 9:** the SIMD16 kernels have 1,506 instructions; the wrapper's SIMD8
  code has 1,866; the SIMD8 code for a required sub-group size of 8 has 1,534.

Both orders use 128 GRF.

### Without `-fp-model=precise`

`FLAGS="" ./build.sh`. IGC picks the same widths, but at nq = 9 its SIMD16 code
is now fast:

| Case | nq = 3: SIMD | GDoF/s | nq = 9: SIMD | GDoF/s |
|---|---|---|---|---|
| 1D range, `kernel_args_restrict` | 32 (IGC) | 7.82 | 16 (IGC) | 8.28 |
| 1D range | 32 (IGC) | 7.81 | 16 (IGC) | 8.30 |
| 2D range | 32 (IGC) | 7.81 | 16 (IGC) | 8.31 |
| 2D range, in a wrapper functor | 16 (IGC) | 8.68 | 8 (IGC) | 7.94 |
| required sub-group size 8 | 8 | 8.39 | 8 | **8.59** |
| required sub-group size 16 | 16 | 8.32 | 16 | 8.26 |
| required sub-group size 32 | 32 | 8.55 | 32 | 5.74 |

So the 1.5× gap at nq = 9 needs both IGC's SIMD16 choice and the precise
floating-point model. At nq = 3, IGC's own SIMD32 code stays slower than the
SIMD32 code for a required sub-group size of 32 (1.09×, against 1.22× with
`-fp-model=precise`).

## The kernel

It is the sum-factorized BK5 kernel (collocated Laplacian) of the CEED
bake-off problems, the variant `sf9c_b1x1` of
[`sycl/bk5_sycl.cpp`](https://github.com/ivan-pi/omp_bk/blob/main/sycl/bk5_sycl.cpp).
`repro.cpp` copies its body verbatim into a macro that every case shares.
Per element:

- one work-group of `round16(nq²)` work-items, one per column of the
  nq × nq × nq points;
- three work arrays in local memory;
- a full 3 × 3 geometric tensor, 9 floats per point, read from global memory.

Each work-item keeps three arrays of nq floats and the 9 tensor values per
point in registers. That register pressure is where the SIMD width matters.

## Files

- `repro.cpp`: the kernel and its cases. It needs only SYCL, with no code from
  the repository. It runs every case on the same pseudo-random data and checks
  that all results are identical.
- `build.sh`: builds `repro_3` and `repro_9` ahead of time for `acm-g10` with
  `-O3 -fp-model=precise`. It dumps IGC's assembly to `asm_3/` and `asm_9/`
  (`IGC_ShaderDumpEnable=1`). The build outputs are not committed
  (`.gitignore`).

```bash
./build.sh
./repro_3
./repro_9
```

`DEVICE` sets another ocloc device, and `FLAGS` replaces `-fp-model=precise`.

## Environment

- GPU: Intel Arc A770 16 GB (acm-g10, PCI ID 8086:56A0)
- Intel oneAPI DPC++/C++ Compiler 2026.1.0 (2026.1.0.20260617)
- ocloc 26.40.40196.15 (`intel-ocloc`), `intel-opencl-icd` and
  `libze-intel-gpu1` 26.40.40196.15, `libigc2` 2.42.14
- Ubuntu 24.04.5 LTS, Linux 6.8.0-136-generic
