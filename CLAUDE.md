# omp_bk

OpenMP implementations of the CEED bake-off kernels BK1, BK3 and BK5; `amx/`
holds the Apple AMX version of BK1 with a software emulator (`amx/amx.h`).
The BK1 kernel lives in `bk1_sumfact.h`, generic in the arithmetic type
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
`-DFF_PLANE_BATCH=n` overrides); `SumFactorizationFFPlaneLoop` is the same
kernel with `teams loop` and nested `loop` constructs, sharing the phase
functions `plane_phase_a..e`. The driver runs and checks all of them; they
must agree bit for bit, as every output is summed in index order.
`bk1_ff_cuda.cuh` is the same kernel as a CUDA kernel (`MassOperatorFF<nq>`,
double storage, float-float inside), compiled only under nvcc;
`float_float.h` is host and device callable for it. `make check_ff_cuda`
runs that kernel on the host through `cuda/cuda_shim.h` against the double
reference (`ok`/`FAIL`); run it after any change to the CUDA header.
`make BK1_ff_cuda` builds the GPU driver `cuda/BK1_ff_cuda.cu` (nvcc, CUDA
13.2+, the cuda::launch runtime API) which times the kernel and checks it
the same way.

## Build and check

- `make` builds BK1, BK3, BK5 and BK1_ff; `make BK1_amx` builds the AMX kernel
  (software-emulated AMX off Apple Silicon); `make bw_test amx_pipe` builds
  the roofline micro-benchmarks in `amx/bench/` (`accel_gemm` is macOS only).
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
