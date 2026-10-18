# nvc++ 25.3: `cache` directive above a `loop worker` faults on the device

Standalone reproducer: `cache_worker.cpp`; `run.sh` builds it with and
without the directive and runs every box size `n` in its own process.

## Report

**Compiler**: nvc++ 25.3-0 linuxarm64 (NVHPC), `-acc=gpu`, target
`-tp neoverse-v2`. **Device**: GH200 (Hopper). **Host**: Grace, aarch64 Linux.

**Construct**: a `parallel loop gang num_workers(W) vector_length(V)` whose
body starts with `#pragma acc cache(readonly: tab[0:ntab])` followed by a
`loop worker` over worker-private arrays and two `loop vector` phases:

```c
#pragma acc parallel loop gang num_workers(W) vector_length(V) copyin(...) copyout(...)
for (int g = 0; g < ngang; ++g) {
    #pragma acc cache(readonly: tab[0:ntab])
    #pragma acc loop worker
    for (int w = 0; w < W; ++w) {
        float s[box];                 // worker-private
        #pragma acc loop vector
        for (int i = 0; i < box; ++i) { s[i] = x[e * box + i] * tab[i % ntab]; }
        #pragma acc loop vector
        for (int i = 0; i < box; ++i) { y[e * box + i] = s[(i + 1) % box] + tab[i % ntab]; }
    }
}
```

The OpenACC specification (3.3, section 2.10) allows the cache directive at
the top of any loop body and places no restriction on the loops nested
below it.

**Expected**: results identical to the build with the directive removed
(`-DNO_CACHE`), which is correct at every `n`.

**Observed**: the build with the directive faults at `n = 4` (box of 64
floats, `W = 16`, `V = 64`):

```
$ ./run.sh
== ./cache_worker_nocache
n =  3 ... ok      (and every n up to 10: ok)
== ./cache_worker
n =  3  W = 32  V =  32  nelmt = 1001  max|diff| = 3.25e-06  ok
Failing in Thread:1
Accelerator Fatal Error: call to cuStreamSynchronize returned error 715 (CUDA_ERROR_ILLEGAL_INSTRUCTION): Illegal instruction
 File: .../acc/repro/cache_worker.cpp
 Function: _Z3runILi4EEii:43
 Line: 85
n = 4: exit status 1
n =  5  W =  8  V = 128  nelmt = 1001  max|diff| = 3.34e-06  ok
n =  6 ... n = 10: ok
```

The kernel this was reduced from (`acc/BK1_acc.cpp`, `BK_KERNEL=worker`,
eight vector phases per worker instead of two) shows the same fault at
`nq = 4` and, in addition, `CUDA_ERROR_MISALIGNED_ADDRESS` at `nq = 7, 8,
9, 10`; without the directive it is correct at every order. The same
directive at the top of a gang loop *without* a worker loop below it works
in every other kernel of that file.

**Workaround**: no cache directive in kernels that use worker-level
parallelism.
