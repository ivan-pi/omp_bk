# The 2021 CM SDK and other old releases

The rest of this skill assumes the current Linux packages: `intel-igc-cm`
from Intel's GPU repository, installed with the driver release it belongs
to (`pipeline.md` §7). This file covers the older releases you may still
meet:

- **CM SDK 20211028:** a tarball with cmc (clang 8.0.1), its own IGC
  1.0.8708 / ocloc / Level Zero loader 1.4, the CPU emulator, HTML docs and
  two examples.
- **GitHub release of intel/cm-compiler:** the last tag is `cmclang-1.0.144`
  (2022). The repository is still active, but its release binaries are
  older than the packages.

Use them only for what the packages can't do (§2), or on a system frozen on
them. Typical signs of old build scripts: `-mcpu=`, `-fcmocl`, `-m64` and a
`--` before the source file.

## Contents
1. Getting and pointing to the SDK
2. What only the 2021 SDK provides
3. Using the 2021 cmc
4. Limitations and bugs (tested)
5. The 2021 SDK docs vs. the GitHub docs
6. The CM runtime host API (`libigfxcmrt`)

## 1. Getting and pointing to the SDK

Intel no longer distributes the 2021 SDK prominently; it unpacks to
roughly 580 MB. If you have the tarball (`cm_sdk_20211028`), unpack it
anywhere (no root needed) and point `CSDK_DIR` at the top directory:
```bash
tar -C <dir> -xf <path-to>/cm_sdk_20211028.tar.*
export CSDK_DIR=<dir>/cm_sdk_20211028
```
The skill's scripts take the SDK only from `CSDK_DIR`:
- `examples/emu/run_emu.sh` and `examples/cmrt/run_cmrt.sh` skip themselves
  without it.
- `scripts/cm_build.sh` and `run_examples.sh` fall back to its cmc when no
  current cmc is on PATH.

Layout of the SDK:

| Path | Contents |
|---|---|
| `usr/bin/cmc` | the compiler |
| `usr/local/lib`, `usr/local/bin/ocloc` | its IGC/ocloc |
| `usr/lib` | emulation libraries, `libclangFEWrapper.so.8`, the Level Zero loader |
| `usr/include/{cm,libcm,shim,igfxcmrt_emu}` | headers |
| `docs/external/` | HTML docs |
| `examples/Sgemm_l0`, `examples/BufferTest_opencl` | Intel's examples |

## 2. What only the 2021 SDK provides

- **CM-runtime kernels (CISA `.isa`).** The legacy CM runtime
  (`libigfxcmrt`, `CmDevice::LoadProgram`) loads a "CISA" container
  (magic `CISA`). Only the 2021 cmc writes it:
  ```bash
  LD_LIBRARY_PATH=$CSDK_DIR/usr/local/lib:$CSDK_DIR/usr/lib \
    $CSDK_DIR/usr/bin/cmc -mcpu=TGLLP -m64 -o k.isa -- k_genx.cpp
  ```
  - It needs the SDK's own libocloc; otherwise it fails with "oclocInvoke
    failed".
  - The current cmc has only `-binary-format ocl|ze`. Both write ELF, which
    `LoadProgram` rejects with -16 (tested on Gen12LP, a UHD 750).
  - `examples/cmrt/run_cmrt.sh hw` builds and runs this route.
- **CPU emulation.** The emulator needs `libcm`, `libshim`, `libshim_l0` and
  `libigfxcmrt_emu`, plus their headers. The kernel is compiled by g++.
  - `examples/emu/run_emu.sh` runs vadd through the OpenCL look-alike shim.
  - `examples/cmrt/run_cmrt.sh emu` runs the CM-runtime API.
  - The upstream emulator project (github.com/intel/cm-cpu-emulation) is
    archived.
- **The media and video documentation** removed upstream (§5).

## 3. Using the 2021 cmc

`$CSDK_DIR/usr/bin/cmc`: clang 8.0.1 plus VC-Intrinsics, target
`genx64-unknown-unknown`.
- Usage: `cmc [options] -- file.cpp`. The `--` is customary because some
  options start with `/`.
- It finds the CM headers by itself; `CM_INCLUDE_DIR` overrides.

| Option | Effect |
|---|---|
| `-mcpu=<BDW\|SKL\|BXT\|KBL\|GLK\|ICL\|ICLLP\|TGLLP\|DG1\|XEHP_SDV>` | target for headers, macros and the built-in finalize step. **No RKL, ADL, DG2 or newer**: compile for the closest older target (TGLLP, or XEHP_SDV for `cm_dpas`/`cm_bfn`) and finalize the SPIR-V for the real device with ocloc |
| `-fcmocl` (= `-binary-format=ocl`) | output for the OpenCL/L0 runtimes (kernel-argument ABI, `[[type(...)]]` annotations). Leave it out only for the CM runtime (`.isa`) |
| `-emit-spirv`, `-emit-llvm -S`, `-E` | SPIR-V, front-end IR, preprocessed source |
| `-m64` | pointer size (`svmptr_t`, `CM_PTRSIZE`) |
| `-g`, `-mCM_no_debug`, `-Qxcm_release`, `-Qxcm_revid=<n>`, `-mCM_printregusage`, `-mdump_asm`, `-cm-printf-spec=ocl\|legacy`, `-fcm-pointer`, `-###` | as named; the finalize-step dumps only work for a target cmc can finalize |

The route that worked for DG2 (tested on an Arc A770):
```bash
cmc -emit-spirv -fcmocl -mcpu=TGLLP -m64 -o k.spv -- k_genx.cpp
ocloc compile -file k.spv -spirv_input -device dg2 \
      -options "-vc-codegen -ftranslate-legacy-memory-intrinsics" -output k -output_no_suffix
```
Its SPIR-V is read by current IGCs: it was tested with IGC 2.42 / NEO
26.40, which still accepts the old SPIR-V dialect. `scripts/cm_build.sh`
switches to these options when the cmc is clang 8; `-m` sets the `-mcpu`.

**Macros:**
- The 2021 cmc predefines `CM_GENX` from `-mcpu` (1200 for TGLLP, 1270 for
  XEHP_SDV), the `CM_GEN12`... platform macros up to XeHP, and part of the
  `CM_HAS_*` set (no `CM_HAS_LSC`).
- `__CM_INTEL_TARGET_*` is **not defined**. Guard code that must build with
  both compilers with `#if defined(__CM_INTEL_TARGET_MAJOR)` (`targets.md`
  §5, idiom 3).
- The macros describe `-mcpu`, not the GPU. `-mcpu=TGLLP` plus `ocloc -device
  dg2` compiles the TGLLP branch for DG2.
- `examples/target_genx.cpp` reports CM_GENX 1200 and IP 0.0.0, and PASSes on the
  legacy path (tested on DG2 and Gen12LP).

## 4. Limitations and bugs (tested)

- **No LSC API:** there is no `cm_lsc.h`. Use the legacy dataport (OWord
  blocks, scattered, `write_atomic`); `examples/lsc_genx.cpp` needs the
  current cmc.
- **`cm_spec_constant` crashes:** `-emit-spirv` segfaults on any kernel that
  calls it (TGLLP and XEHP_SDV alike). `-emit-llvm` works, but oneAPI's
  `llvm-spirv` rejects cmc's `genx64` IR, so there's no workaround. Use `-D`,
  templates or scalar arguments (`language.md` §8).
- **No run-time `-cmc` source builds:** the SDK's `libclangFEWrapper.so.8` is
  interface 10, and current NEO expects 11 ("incompatible clangFEWrapper
  interface: expected = 11, loaded = 10").
- **`[[type("image2d_media_block_t")]]` is ignored**, with only a warning.
  The argument silently becomes a buffer and `clSetKernelArg` fails with
  -38. Use `image2d_t`, which means a media-block image in both compilers.
- **SLM handle arithmetic is silently dropped:**
  `cm_slm_block_write(slm + lid * 16, 0, v)` compiles but every thread
  writes slot 0. Pass the slot offset as the offset argument, which is
  correct for every cmc (`memory.md` §4).
- **`cmtl.h`:** a 4-element `cm_vector` fails with "select offset cannot be
  negative"; 16 elements work. Use a const-array initializer instead.
- **`cm_dpas`:** only with `-mcpu=XEHP_SDV`, and ExecSize is hard-coded to
  8. Finalized for dg2 it still gives a native `dpas.8x8`.
- **`[[type("svmptr_t")]]` warning:** cmc warns that the annotation is
  ignored. It is harmless: `clSetKernelArgSVMPointer` works.
- **Bundled runtime:** the SDK's IGC 1.0.8708 and Level Zero loader 1.4 are
  for its own tools only. Never put `usr/local/lib` on a global library
  path.

## 5. The 2021 SDK docs vs. the GitHub docs

The SDK's HTML (`docs/external/`, built 2021-10-26) has the spec, the
compiler guide and the OpenCL runtime notes, but **no tutorial**. Compared
with the GitHub spec by section headings:

- **Only on GitHub (newer):**
  - the LSC data-port API (4.5: `cm_load`/`cm_store`/`cm_prefetch`,
    `cm_ptr_*`, SLM `cm_load_slm`..., 2D and descriptor-based 2D blocks,
    typed LSC, `Fence`, cache-hint tables);
  - the XMX section (`cm_dpas`, `cm_dpasw`, `cm_bdpas`);
  - named barriers `cm_nbarrier_*` (PVC+), `cm_raw_send`/`cm_raw_sendg`;
  - the address-spaces chapter 7;
  - FP8/TF32/MXFP conversions (`cm_bf8_cvt`, `cm_hf8_cvt`, `cm_srnd*`,
    `cm_mxfp_*`);
  - `cm_lfsr`, `cm_bfn`, `cm_assert` and `cm_get_tileid`.

  The 2021 headers do contain `cm_dpas` and `cm_bfn` for `-mcpu=XEHP_SDV`,
  even though the 2021 spec has no section on them.
- **Only in the 2021 spec (removed upstream):**
  - the AVS sampler;
  - Video Analytics: 2D/1D/1-pixel convolve, min-max filter, erode, dilate,
    centroid, LBP, flood fill, correlation search;
  - the VME interface for Gen6 to Gen10 HEVC, with its access-macro
    appendix;
  - Transpose Read;
  - extended math: `cm_fmod`, `cm_floor`, `cm_ceil`, `cm_atan2`;
  - per-generation tags such as `{Gen7+}`.

  The CM template library chapter moved to GitHub's separate `cmtl.rst`.
- **The same in both**, apart from wording: types, the select family, SIMD
  control flow, the legacy data-port API, SLM, SVM and inline assembly.

## 6. The CM runtime host API (`libigfxcmrt`)

The legacy tasking layer (`examples/cmrt/hello_world.cpp`, a
complete version of the flow in Intel's "Precious Metal" article):
```cpp
CmDevice *dev;  UINT ver;   ::CreateCmDevice(dev, ver);
CmProgram *prog;            dev->LoadProgram(isa.data(), isa.size(), prog);   // .isa from cmc
CmKernel *k;                dev->CreateKernel(prog, CM_KERNEL_FUNCTION(hello_world), k);
CmThreadSpace *ts;          dev->CreateThreadSpace(w, h, ts);       // kernel: get_thread_origin_x/y()
CmBuffer *buf; SurfaceIndex *idx;  dev->CreateBuffer(bytes, buf);  buf->GetIndex(idx);
k->SetKernelArg(0, sizeof(w), &w);  k->SetKernelArg(1, sizeof(SurfaceIndex), idx);
CmTask *task;               dev->CreateTask(task);  task->AddKernel(k);
CmQueue *q; CmEvent *ev;    dev->CreateQueue(q);  q->Enqueue(task, ev, ts);
ev->WaitForTaskFinished();  buf->ReadSurface(host_ptr, ev);  ::DestroyCmDevice(dev);
```
- Every call returns `CM_SUCCESS` (0) or a negative code; check each one.
- **Device selection:** `CreateCmDevice(dev, ver)` without a VADisplay
  can fail with `CM_NULL_POINTER` (-90) on a system with several GPUs
  (seen with an Arc A770 next to a UHD 750). Enumerate
  with `GetCmSupportedAdapters` and `QueryCmAdapterInfo(a, Description |
  DeviceId, ...)`, then use `CreateCmDeviceFromAdapter(dev, ver, a)`. The
  alternative is `CreateCmDevice(dev, ver, vaGetDisplayDRM(fd))` on a chosen
  `/dev/dri/renderD*` node. Adapters the runtime doesn't support (e.g. a
  DG2 card, see the next point) fail with -90, so try each adapter in turn
  and keep the first that succeeds.
- **The media driver's CM runtime does not support DG2/Arc.** Its headers
  stop at Gen12 TGL/DG1. On Arc and newer, launch CM kernels through
  OpenCL/Level Zero/SYCL (layer 4a) or move them to ESIMD.
- **Kernel printf on the GPU** needs `dev->InitPrintBuffer()` before the
  enqueue and `dev->FlushPrintBuffer()` afterwards. Otherwise nothing is
  printed and no error is reported.
- **Builds need the 2021 SDK** (`legacy-sdk.md` §2): the `.isa` is a CISA container only its cmc
  writes (today's cmc writes ELF, rejected with -16); emulation links its `-ligfxcmrt_emu -lcm`.
  Link a hardware host with `-ligfxcmrt -lva`. No `[[type("...")]]` annotations are needed.
- `examples/cmrt/run_cmrt.sh [emu|hw|both]` builds and checks both modes with `CSDK_DIR` set.
