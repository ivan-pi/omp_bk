# tinytc: kernels over 64 KB of local memory fail with an undiagnosed runtime error

Minimal reproducer: a tinytc kernel whose local memory (`alloca ... local`)
exceeds the device's local memory per work-group. On an Arc A770 (DG2,
Alchemist), the limit is 65536 bytes. One more float than that makes kernel
creation fail with only:

```
Source name unavailable:0.0:
tinytc error 18: Error occured in compute runtime
```

The message doesn't say what went wrong. The Level Zero call underneath is
`zeModuleCreate`, which returns `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` and an
empty build log. That error is misleading: device memory (16 GB) is not
exhausted; the kernel's local memory exceeds the per-work-group limit.

How we found it: the BK5 kernel `bk5.ir` (B = 16 elements per work-group) needs
2 × 16 × nq³ floats of local memory. At nq = 9 that is 93 KB, and every run
failed with the error above. The BK1 kernel at nq ≥ 9 fails the same way:
its two largest work arrays, live in the same step, take 83 KB there.

## Environment

- GPU: Intel Arc A770 (acm-g10), Level Zero driver 1.15.38646+6
  (`libze-intel-gpu1` 26.22.38646.6, `intel-opencl-icd` 26.22.38646.6,
  `libigc2` 2.36.3)
- tinytc v0.5.0-0-ac6cbc7 (`/opt/tinytc/0.5.0`)
- Intel oneAPI DPC++/C++ Compiler 2026.1.0 (2026.1.0.20260617)
- Linux 6.8.0-136-generic (Ubuntu 24.04)

## Files

- `repro.cpp`: for each N on the command line, compiles the kernel below for
  the SYCL GPU device (`tinytc::create_core_info(device)` →
  `compile_to_spirv_and_assemble` → `create_kernel_bundle`), launches one
  work-group of 16 work-items and checks `out[0] == 1`:

  ```
  func @slm(%out: memref<f32x?>)
  attributes{subgroup_size=16,work_group_size=[16,1]}
  {
      %buf = alloca : memref<f32x$N,local>
      %c0 = constant 0 : index
      %x = constant 1.0 : f32
      store %x, %buf[%c0]
      barrier.local
      %y = load %buf[%c0] : f32
      store %y, %out[%c0]
  }
  ```
- `build.sh`: builds `repro` with icpx and tinytc from `/opt/tinytc/latest`
  (or `$TINYTC_ROOT`).
- `run.sh`: runs `repro` in the oneAPI environment.
- `run_output.txt`, `ze_trace_16385.txt`, `run_output_zero.txt`: the output
  below.

## Steps

```bash
./build.sh
./run.sh 4096 16384 16385
```

Output (`run_output.txt`):

```
device: Intel(R) Arc(TM) A770 Graphics, driver 1.15.38646+6
local memory per work-group (sycl::info::device::local_mem_size): 65536 bytes
tinytc v0.5.0-0-ac6cbc7
N = 4096 floats (16384 bytes of local memory): ran, out[0] = 1 (ok)
N = 16384 floats (65536 bytes of local memory): ran, out[0] = 1 (ok)
N = 16385 floats (65540 bytes of local memory): Source name unavailable:0.0:
tinytc error 18: Error occured in compute runtime
```

The Level Zero error, from the loader's tracing (`ze_trace_16385.txt`):

```bash
ZE_ENABLE_VALIDATION_LAYER=1 ZE_ENABLE_PARAMETER_VALIDATION=1 ZEL_ENABLE_LOADER_LOGGING=1 \
    ZEL_LOADER_LOGGING_LEVEL=trace ZEL_LOADER_LOG_CONSOLE=1 ./run.sh 16385
```

```
[trace] zeModuleCreate(hContext, hDevice, desc, phModule, phBuildLog)
[trace] ERROR (ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) in zeModuleCreate(...)
[trace] zeModuleBuildLogGetString(hModuleBuildLog, pSize, pBuildLog)
[trace] zeModuleBuildLogGetString(hModuleBuildLog, pSize, pBuildLog)
[trace] zeModuleBuildLogDestroy(hModuleBuildLog)
tinytc error 18: Error occured in compute runtime
```

The same SPIR-V builds without error offline:

```bash
ocloc compile -spirv_input -file <kernel.spv> -device acm-g10
```

ocloc doesn't check the kernel's local memory against the device limit.

## Expected

- **tinytc:** reject the kernel at compile time with a message naming the
  local memory requested and the device limit. tinytc has the device; SYCL
  reports `local_mem_size` = 65536. Failing that, pass on the reason from the
  runtime instead of a generic status 18 with an empty message.
- **Level Zero / compute runtime:** an error that names the cause, for example
  `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE` or `..._OUT_OF_RESOURCES` with the
  local memory size in the build log, rather than
  `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` and an empty log.

## Related, seen while making this reproducer

- **Zero-size local array:** `./run.sh 0` crashes tinytc instead of giving a
  diagnostic (`run_output_zero.txt`):
  ```
  <memory>:6.12-40: Internal compiler error (Stack required but not allocated)
      %buf = alloca : memref<f32x$N,local>
  tinytc error 5: Internal compiler error
  ```
- **No DG2 target:** tinytc 0.5.0's `intel_gpu_architecture` list has tgl,
  pvc, bmg, bmg_g31, lnl, ptl_h and ptl_u, but not DG2 (acm). The `tinytc`
  CLI can't target this GPU (`-d acm-g10` → "Invalid arguments passed to
  function"). The runtime path works from the queried device properties.
- **f32 cooperative matrices use no XMX:** they are lowered to ordinary FMAs
  (no `dpas`, on DG2 or with the CLI's default pvc target). That is expected:
  DG2's XMX has no fp32 or tf32 mode. Noted here because the IR's
  cooperative-matrix syntax suggests otherwise.
