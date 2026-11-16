# omp_bk

The CEED bake-off kernels BK1, BK3 and BK5, organized by approach:
`openmp/` (drivers for host or offload; GPU kernel variants in
`openmp/offload/`), `sycl/`, `tinytc/` (driver and `ir/` kernels), `amx/`
(the Apple AMX version of BK1 with a software emulator, `amx/amx.h`),
`roofline/` (bandwidth and peak micro-benchmarks), `include/` (headers
shared across approaches), `scripts/` (Intel GPU build, run, `study/`,
`plot/`, `profile/`) and `results/<machine>/`. File names carry the kernel
(`bk1_*`, `bk5_*`).

The BK1 kernel lives in `openmp/bk1_sumfact.h`, generic in the arithmetic type
(float or double); `BK1.cpp` runs it in float. `BK1_ff.cpp` holds its own
float-float kernels with the same interface in double: values are split into
hi/lo float pairs inside the kernel, the work arrays are separate hi and lo
float arrays, and the TwoSum/TwoProd helpers come from `float_float.h`, a
private header of that driver only. `SumFactorizationFF` runs one element
per thread, `SumFactorizationFFTeams` one element per team with the work
arrays at team scope and a `parallel for collapse(3)` per step (team size
from the constexpr `ff_team_size`), and `SumFactorizationFFPlane<nq, batch>`
a batch of elements per team of `batch * nq * nq` threads, each owning a
register column, with five phases (default batch from `ff_plane_batch`,
`-DFF_PLANE_BATCH=n` overrides). The driver runs and checks all of them;
they must agree bit for bit, as every output is summed in index order.

## Build and check

- CMake, with presets. `cmake --preset host && cmake --build build/host`
  builds BK1, BK3, BK5, BK1_ff, BK1_amx (software-emulated AMX off Apple
  Silicon) and the roofline micro-benchmarks (`accel_gemm` is macOS only).
  The Intel GPU binaries (one order each, named by p: `BK1_4`,
  `bk1_offload_4`, `bk5_sycl_6`, `bk_tinytc`, ...) are built with
  `scripts/build.sh <target>...` into `build/a770` (memory-capped, one
  compile at a time), never while a GPU run is going on. Backends are CMake
  options: `BK_ENABLE_OPENMP`, `BK_ENABLE_SYCL` (both on, both from icpx)
  and `BK_ENABLE_TINYTC` (off by default, on in the `a770` preset); keep any
  new toolchain beyond icpx (e.g. C for Metal) behind its own default-off
  option that fails at configure time when asked for and missing.
- `BK1_ff` compares double, float and float-float on the same data and ends
  with an `ok`/`FAIL` check of the float-float result (exit status 1 on
  failure). Run it for p = 1..8, with and without `BK_RANDOM=1`, after any
  change to `float_float.h` or `bk1_sumfact.h`. Never build with
  `-ffast-math`; the header rejects it, and anything subtler that breaks IEEE
  rounding shows up as `FAIL`.
- `amx/scripts/validate.sh` checks the AMX kernel against the serial
  reference on random data; the emulator also enforces the 128-byte
  alignment of pair/quad loads and stores, so run it after any layout change.

## C++ style

- Every `for`, `if` and `else` body is wrapped in `{ }`, including nested
  loops and single-statement bodies.
- One statement per line; do not put several statements on one line
  (a one-statement function body may stay on the function's line).
- Use pre-increment (`++i`) in loop counters.
- State preconditions and invariants as `static_assert` where the value is a
  compile-time constant and as `assert` otherwise; keep runtime asserts at
  batch granularity or coarser, never inside the per-plane hot loops
  (the benchmark builds without `-DNDEBUG`).
- OpenMP regions take `if(parallel)` from the options so the same binary
  runs single-threaded (`BK_PARALLEL=0`); a flag only used by a pragma is
  marked `[[maybe_unused]]` for builds without OpenMP.
