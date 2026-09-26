# omp_bk

OpenMP implementations of the CEED bake-off kernels BK1, BK3 and BK5; `amx/`
holds the Apple AMX version of BK1 with a software emulator (`amx/amx.h`).

## Build and check

- `make` builds BK1, BK3 and BK5; `make BK1_amx` builds the AMX kernel
  (software-emulated AMX off Apple Silicon).
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
