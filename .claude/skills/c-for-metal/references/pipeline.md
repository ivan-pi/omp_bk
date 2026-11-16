# C for Metal compilation pipeline, drivers and options

The results in this file were tested with the current packaged cmc
(`intel-igc-cm` 1.0.225, its matched IGC 2.11 / ocloc 25.18), a NEO 26.40
runtime and oneAPI 2026.1, on an Arc A770 (DG2, XeHPG 12.55.8) and a UHD 750
(Gen12LP, RKL 12.1.0). The 2021 CM SDK and the 2022 GitHub release
are covered in `legacy-sdk.md`.

## Contents
1. Stages at a glance
2. cmc (front end): driver, outputs, options
3. SPIR-V: what it contains, how to read it
4. ocloc / IGC vector-compute backend: options, dumps
5. Run-time compilation paths (OpenCL, Level Zero, SYCL)
6. Inspecting performance: vISA, asm, register use
7. Installing cmc: packages, IGC pairing, version checks

---

## 1. Stages

```
 kernel_genx.cpp (CM = C++17 subset + vector/matrix + intrinsics, <cm/cm.h> implicit)
   │  cmc -march=<device>  [clang 11 front end, target genx64]
   │      -E            → preprocessed .i
   │      -emit-llvm -S → LLVM IR (.ll), unoptimized (optnone), CM ops as llvm.genx.* intrinsics
   │      -emit-spirv   → SPIR-V with SPV_INTEL_vector_compute       ◄── portable artifact
   │      (default)     → zebin via the ocloc of its release
   ▼
 kernel.spv
   │  IGC "VC" backend (vector compute), invoked by:
   │      ocloc compile -spirv_input -options "-vc-codegen"           (offline, AOT)
   │      clCreateProgramWithIL + clBuildProgram("-vc-codegen")       (JIT, OpenCL)
   │      zeModuleCreate(IL_SPIRV, pBuildFlags="-vc-codegen")         (JIT, Level Zero)
   │      SYCL kernel_compiler(source_language::spirv, "-vc-codegen") (JIT, SYCL)
   │   inside IGC:  SPIR-V reader → GenX LLVM passes (lowering, baling/legalization,
   │                region collapsing, vector decomposition, coalescing, ...) → vISA
   │                → vISA finalizer (register allocation, scheduling, SWSB) → Gen ISA
   ▼
 kernel.bin (zebin ELF: .text.<kernel>, .ze_info, .spv copy, build options)
   │  clCreateProgramWithBinary / zeModuleCreate(NATIVE)
   ▼
 GPU
```

The paper's middle-end design: CM regions are modelled in SSA LLVM IR by
two intrinsics. `llvm.genx.rdregion` reads a strided region of a vector into
a new value. `llvm.genx.wrregion` returns a new vector with a region
replaced. `v.select<4,2>(1) = b` becomes a `wrregion` with vstride 0,
width 4, stride 2 and byte offset 4. Baling then folds rdregion/wrregion
into the source and destination operands of one vISA instruction, so a
`select` costs nothing when it fits a Gen region `<V;W,H>`.

## 2. cmc

`cmc` from the `intel-igc-cm` package (clang 11.1 plus VC-Intrinsics, target
`genx64-unknown-unknown`; install: §7). It finds the CM headers by itself
(`<cm/cm.h>` is included implicitly); don't pass `-I` to the package's
`usr/include`, or its headers count as user code and `-Werror,-Wgen-target`
rejects their `double` uses on DG2.

```bash
cmc -march=dg2 -emit-spirv -o k.spv k_genx.cpp     # portable SPIR-V (finalize with -vc-codegen)
cmc -march=dg2 -o k.bin k_genx.cpp                 # SPIR-V + ocloc in one step: zebin
cmc -march=dg2 -emit-llvm -S -o k.ll k_genx.cpp    # front-end IR
cmc -march=rkl -E -o k.i k_genx.cpp                # preprocess (target macros for rkl)
```

| Option | Effect |
|---|---|
| `-march=<device>` | target: an ocloc device name (`dg2`, `acm-g10`, `rkl`, `tgllp`, `mtl-h`, `bmg`, ...) or an IP version (`12.55.8`). Sets the target macros (`targets.md`) and the finalize target. `rkl` is not `tgllp` |
| `-emit-spirv`, `-emit-llvm -S`, `-E` | SPIR-V, front-end LLVM IR (unoptimized; `-O2` is ignored), preprocessed source |
| `-binary-format ocl\|ze` | runtime ABI of the output (`ocl`, the default, serves OpenCL, Level Zero and SYCL). There is no CM-runtime (CISA) output any more: `legacy-sdk.md` |
| `-D`, `-I` | as in clang. **The language level is fixed at C++17** (`__cplusplus` = 201703L); any `-std=` is rejected |
| `-Qxcm_doubleGRF`, `-Qxcm_register_file_size=<128\|256\|auto>` | 256 GRFs per thread (the ocloc equivalent is `-double-GRF`) |
| `-g`, `-mCM_no_debug`, `-Qxcm_release` | debug info control |
| `-mCM_printregusage`, `-Qxcm_print_asm_count`, `-mdump_asm` | statistics/dumps of the built-in finalize step; the IGC dumps (§4) give more |
| `-mCM_no_vector_decomposition`, `-mCM_jit_option<v>` | backend tuning knobs passed through |
| `-###` | print the internal commands |

Old-style options are accepted but unnecessary: `-fcmocl` (ignored with a
warning), `-m64`. `-mcpu=` and the `--` before the file belong to the 2021
cmc.

Predefined macros: `__CM`, `__CMC`, `__CM_INTEL_TARGET_MAJOR/MINOR/REVISION`,
`__CM_INTEL_TARGET_ID`, the `CM_HAS_*` feature macros (`CM_HAS_LSC`,
`CM_HAS_DPAS`, `CM_HAS_BF16`, `CM_MAX_SLM_SIZE`, ...) and the deprecated
`CM_GENX`; all in `targets.md`.

The front-end IR for `vadd` contains calls such as
`@llvm.genx.group.id.x()`, `@llvm.genx.local.id.v3i32()`,
`@llvm.genx.oword.ld.v32i8(i32 0, i32 %bti, i32 %off)` and
`@llvm.genx.oword.st...`. Every CM intrinsic is visible at this level, which
makes the IR the right place to check what a CM construct turned into
before the backend optimizes it.

## 3. SPIR-V

```
Capability VectorComputeINTEL, VectorAnyINTEL, Kernel, Linkage, ...
Extension "SPV_INTEL_vector_compute"
EntryPoint Kernel "vadd"            ExecutionMode ... SubgroupSize-like 1 (SIMD1)
Decorate <fn> VectorComputeFunctionINTEL
Decorate <fn> LinkageAttributes "llvm.genx.oword.ld.v32i8" Import   ← GenX intrinsics stay as imported functions
Decorate <param> FuncParamIOKind 0
```
Disassemble it with the oneAPI copy of the translator:
`/opt/intel/oneapi/compiler/latest/bin/compiler/llvm-spirv -to-text k.spv -o k.spvasm`.
It does not depend on the device: the same `.spv` ran unchanged on DG2
(Arc A770) and Gen12LP (UHD 750).

## 4. ocloc and the IGC VC backend

```bash
ocloc compile -file k.spv -spirv_input -device dg2 \
      -options "-vc-codegen [backend options]" -output k -output_no_suffix [-out_dir d] [-q]
ocloc compile -file k.cpp -device dg2 -options "-cmc [-DNAME=v ...]"   # from CM source: needs
      # the CM front end (libclangFEWrapper.so from intel-igc-cm) AND the IGC of its release (§7)
```
Device names: `dg2` (= `acm-g10`, 12.55.8), `tgllp`, `rkl`, `adl-s`, `mtl-u`, `mtl-h`,
`arl-h`, `bmg`, `lnl`, `pvc`, ... (`ocloc compile --help` lists them).

`-vc-codegen` backend options (cmocl guide, plus the ones tested):

| Option | Effect |
|---|---|
| `-ftranslate-legacy-memory-intrinsics` | rewrite legacy dataport messages (oword/scattered/atomic) for the target. **Required on DG2 for legacy `write_atomic`** (it fails at run time without it). Only valid for LSC targets: Gen12LP fails to build with it |
| `-double-GRF` (L0 alias `-ze-opt-large-register-file`) | 256 GRFs per thread (half the threads); tested OK on DG2 |
| `-Xfinalizer '<opts>'` | pass options to the vISA finalizer, e.g. `'-printregusage -noLocalSplit'` (accepted by ocloc, but `-printregusage` printed nothing to stdout; read `numGRF`/spills from the IGC `.asm` dump instead) |
| `-no-vector-decomposition` | keep large vectors whole |
| `-fno-jump-tables` | no SWITCHJMP |
| `-vc-use-plain-2d-images` | `image2d_t` means a sampler image, not a media block |
| `-vc-disable-non-overlapping-region-opt` | debugging aid |
| `-g` | debug info |

Restrictions: no global work offset; surfaces at most 4 GB (stateful).

**Dumps** (any path that runs IGC: ocloc, clBuildProgram, zeModuleCreate):
```bash
IGC_ShaderDumpEnable=1 IGC_DumpToCurrentDir=1 ocloc compile ...   # or run the host program
```
This writes `VC_<hash>_*` files: `after_spirv_reader.ll`, `optimized.ll`
(GenX LLVM after the middle end), `final.ll`, `<kernel>.visaasm` (vISA),
`<kernel>.asm` (final Gen assembly, with `.thread_config numGRF=...`,
`.instCount`, ALU/send statistics and spill information), `.zeinfo`,
`options.txt`. Without `IGC_DumpToCurrentDir` they go to IGC's
default dump directory (`IntelIGC/<app>/` under the system temporary
directory). `scripts/cm_build.sh -k` collects all of them.

`ocloc disasm` on the zebin produced only `sections.txt` when tested. Use the IGC
dumps to get assembly.

## 5. Run-time paths

| Path | Call | Options | Status (tested on DG2 and Gen12LP) |
|---|---|---|---|
| OpenCL, SPIR-V | `cl::Program(ctx, il_bytes)` (= `clCreateProgramWithIL`) + `build({dev}, ...)` | `-vc-codegen` (+ `-ftranslate-legacy-memory-intrinsics` on LSC GPUs) | PASS (A770, UHD 750) |
| OpenCL, binary | `cl::Program(ctx, {dev}, binaries)` (= `clCreateProgramWithBinary`) | — | PASS |
| OpenCL, CM source | `cl::Program(ctx, source)` (= `clCreateProgramWithSource`) | `-cmc [-D...]` | PASS with `intel-igc-cm` 1.0.225 + its matched runtime (§7), on an A770 and a UHD 750 |
| SYCL (L0 and OpenCL backends), SPIR-V | `create_kernel_bundle_from_source(ctx, source_language::spirv, bytes)` + `build(..., build_options{"-vc-codegen"})` | `-vc-codegen` | PASS with the matched runtime (`ext_oneapi_can_compile(spirv)` wrongly returns false; ignore it) |
| SYCL (L0 and OpenCL backends), binary | Level Zero `zeModuleCreate(ZE_MODULE_FORMAT_NATIVE)` or OpenCL `clCreateProgramWithBinary`, wrapped with `sycl::make_kernel_bundle<backend, executable>`; kernels from `zeKernelCreate`/`clCreateKernel` + `sycl::make_kernel` (`ext_oneapi_get_kernel` only works on source-built bundles) | | PASS on both backends, A770 and UHD 750 (`cmsycl::load_program`) |
| SYCL (L0 and OpenCL backends), CM source | `create_kernel_bundle_from_source(ctx, source_language::opencl, text)` + `build(..., build_options{"-cmc"})` | `-cmc` | PASS with the matched runtime on both backends, A770 and UHD 750 (`examples/host_sycl.cpp k_genx.cpp`) |
| Level Zero, SPIR-V | `zeModuleCreate` with `ZE_MODULE_FORMAT_IL_SPIRV`, `pBuildFlags = "-vc-codegen"` | | this is what the SYCL L0 path does |
| Level Zero, binary | `ZE_MODULE_FORMAT_NATIVE` | | |
| Level Zero, CM source | none: `zeModuleCreate` takes only SPIR-V or native binaries | | compile with ocloc (offline, or `libocloc`'s `oclocInvoke` at run time, as SYCL does) |

### Run-time CM source builds (`-cmc`): what they are and where they work

The host hands the **CM source text itself** (the `*_genx.cpp` file, not
SPIR-V) to the runtime, with build options `-cmc` plus any `-D...`, as an
OpenCL C program would be built. It is the CM equivalent of OpenCL's
textual substitution at `clBuildProgram` time.
- **Intel GPU drivers only.** `-cmc` is not an OpenCL feature: it is an
  option of Intel's compute runtime (NEO: the OpenCL GPU driver, its Level
  Zero driver and `ocloc`). Given `-cmc`, NEO passes the source to the CM
  front end, `libclangFEWrapper.so`, found through the library path. The
  front end produces vector-compute SPIR-V, and IGC finalizes that.
- **Other runtimes:** another vendor's OpenCL, or Intel's CPU OpenCL
  runtime, would treat the text as OpenCL C and fail. CM has no CPU path
  except the emulator.
- **Requirements:** the front end from `intel-igc-cm`, **and** an IGC from the same driver release (the
  front end's SPIR-V fails a newer IGC's reader, such as 2.42's, exactly like cmc's). The CM
  headers are built into the front end, so no include path is needed.
- **OpenCL:** `clCreateProgramWithSource` + `clBuildProgram(..., "-cmc -DCHUNK=128")`.
  `cmocl::load_program` does this for `.cpp` files.
- **SYCL:** `kernel_compiler` with `source_language::opencl` and
  `build_options{"-cmc"}`. DPC++ compiles "OpenCL C" source by calling
  `ocloc -spv_only -options "-cmc" -file main.cl`, and then builds the
  SPIR-V on the selected backend. That is why it works on **Level Zero**
  too, although Level Zero itself has no source input. It depends on the
  `libocloc.so` that DPC++ loads, which must be the matched one (a
  system-wide install has it; with the user-space install, `cm_run` puts it
  first on the library path).
- **Offline:** `ocloc compile` (`ocloc-cm` for the user-space install) -file k_genx.cpp -device dg2 -options "-cmc"`
  gives a `.bin` that runs on newer drivers too.
- **Failure messages:**
  - `AdaptorCM: could not load FEWrapper` means no `libclangFEWrapper.so`
    on the library path;
  - `incompatible clangFEWrapper interface: expected = 11, loaded = 10`
    means an older front end, such as the 2021 SDK's (`legacy-sdk.md`);
  - `spirv_read_verify: verify Module failed` means the right front end
    with an IGC from another release.

Level Zero sketch (not separately compiled and tested; SYCL covers the same call):
```cpp
ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC};
md.format = ZE_MODULE_FORMAT_IL_SPIRV;
md.inputSize = spv.size();  md.pInputModule = (const uint8_t *)spv.data();
md.pBuildFlags = "-vc-codegen";
zeModuleCreate(ctx, dev, &md, &module, &buildLog);
ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC}; kd.pKernelName = "vadd";
zeKernelCreate(module, &kd, &kernel);
zeKernelSetGroupSize(kernel, 1, 1, 1);                       // threads per group
zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &usm_a); // buffer_t ← USM pointer
ze_group_count_t groups = {nthreads, 1, 1};
zeCommandListAppendLaunchKernel(cmdlist, kernel, &groups, nullptr, 0, nullptr);
```

Dispatch is always **SIMD1**: each work-item is one CM thread, all 32
channels enabled. The global size counts threads and the local size counts
threads per group.

## 6. Performance inspection checklist

1. `cm_build.sh -k`, then read `igc_dump/*_<kernel>.asm`:
   - `numGRF` and spill notes. Any spill (`scratch`, `spill` in the asm) needs fixing first.
   - Count of `send` instructions (memory messages) and their widths: one
     `send` per OWord/media block is what you want.
   - `mov` instructions with regions: selects that did not fold into a
     consumer.
2. Look at `optimized.ll` to see whether `rdregion`/`wrregion` survived as
   expected, for example after a `select` with a variable offset (indirect).
3. Use larger blocks per message: 2D media block (up to 64 B x 4 rows ...
   4 B x 64 rows), 128 B OWord blocks, and LSC blocks up to 64 DWords (DG2 and newer).
4. GRF budget: 128 x 32 B = 4 KB (DG2). Use `-double-GRF` when the working
   set needs it and occupancy matters less.
5. Time a kernel with OpenCL events (`CL_QUEUE_PROFILING_ENABLE`) or SYCL
   events. The first JIT run includes compilation; use AOT `.bin` files to
   take that out.

## 7. Installing cmc: packages, IGC pairing, version checks

### What exists

- **Packages (use these):** Intel's GPU apt repository ships `intel-igc-cm`
  (description "C for Metal Compiler -- CM Frontend lib") with every driver
  release. It contains `/usr/bin/cmc`, `libclangFEWrapper.so.11` (clang
  11.1, front-end interface 11) and the headers, including `cm_lsc.h`,
  `lsc/block2d.h` and `cm_spec_constant.h`. The newest version is
  `1.0.225.54093`; the 1.0.176 builds are for the older IGC 1.x.
- **Version listings:** the per-release package lists on
  https://dgpu-docs.intel.com show which `intel-igc-cm` belongs to which
  driver release (e.g. `legacy/release-notes/packages.html?release=LTS+2523.40&os=Ubuntu+24.04`).
  Intel moves these pages, so start from the site root.
- **Not these:** the GitHub release of https://github.com/intel/cm-compiler
  (tag `cmclang-1.0.144`, 2022; the repository itself is still active) and
  the 2021 CM SDK are older than the packages: `legacy-sdk.md`.
- **Distribution and PPA drivers:** drivers packaged by a Linux
  distribution or a third-party PPA often do **not** come with
  `intel-igc-cm`; check before relying on it.

### The pairing rule (tested)

A cmc's SPIR-V must be finalized by the **IGC of its own driver release**:

| cmc | IGC that finalizes | Result |
|---|---|---|
| `intel-igc-cm 1.0.225.54093` | IGC 2.11.12 + ocloc 25.18 (driver build 1146) | works, all examples PASS (tested on an A770 and a UHD 750) |
| `intel-igc-cm 1.0.225.54093` | IGC 2.42.14 (compute-runtime 26.40) | **fails** for every kernel: `spirv_read_verify: verify Module failed` |

Device **binaries** cross releases: a zebin built with ocloc 25.18 loads
and passes on the 26.40 runtime. Only finalizing SPIR-V at run time (the IL
path, spec constants set at run time, `-cmc` source builds) needs the
matched runtime.

### Install

**System-wide, from Intel's repository (the normal case).** Add the
repository as dgpu-docs.intel.com describes for your distribution, then
install the compiler with the runtime of the **same release**:
```bash
sudo apt install intel-igc-cm intel-opencl-icd libze-intel-gpu1 libigc2 libigdfcl2 intel-ocloc
```
Then `cmc`, `ocloc` and every OpenCL / Level Zero / SYCL program use the
matched pair, and none of the wrappers below are needed. Upgrade them
together; if another source (a PPA, the distribution) also provides the
driver, pin Intel's repository so IGC isn't upgraded past the compiler.

**User space, when the system driver comes from a source without a
matching `intel-igc-cm`** (e.g. a distribution or a PPA):
```bash
<skill-dir>/scripts/install_cm_toolchain.sh   # default prefix ~/opt/intel-cm-<build> (~100 MB download)
source <prefix>/env.sh                        # PATH += cmc, ocloc-cm, cm_run wrappers
```

The script fetches `intel-igc-cm`, `libigc2`, `libigdfcl2`, `intel-ocloc`,
`intel-opencl-icd` and `libze-intel-gpu1` of one driver build (`-b 1146`) from
`repositories.intel.com/gpu/ubuntu` (`dists/noble/unified`) and unpacks
them with `dpkg-deb -x`. It then writes three wrappers (**this skill's own
scripts, not Intel tools**: the packages contain `cmc` and `ocloc-25.18.1`), so the matched
libraries are seen only by commands that ask for them. IGC 2.11 loaded
under a newer runtime (e.g. 26.40) fails, so don't put it on a global
`LD_LIBRARY_PATH`.
- `cmc`: the front end, using its own ocloc for `-o k.bin`.
- `ocloc-cm`: ocloc 25.18 with IGC 2.11.
- `cm_run <host> ...` (alias `cm_ocl`): runs a host program on the matched
  OpenCL **and** Level Zero drivers. It sets `OCL_ICD_VENDORS` to a
  private `.icd`, `ZE_ENABLE_ALT_DRIVERS` to the 25.18 `libze_intel_gpu.so.1`
  (the Level Zero loader's switch for an alternative driver) and the
  library path, which covers the matched IGC, `libocloc.so` and the CM
  front end. SYCL programs and `sycl-ls` see the matched drivers under it
  (`sycl-ls` reports `[1.6.33578+15]`).

`cm_build.sh` and `run_examples.sh` use `ocloc-cm` and `cm_run` when they are
on PATH, and plain `cmc`/`ocloc` otherwise. Tested with it: all examples
pass on an Arc A770 (DG2); on a UHD 750 (Gen12LP) everything passes except
LSC, which is skipped because Gen12LP has no LSC.

### Notes on the current cmc

- **Tested and working:** `-march=dg2`/`rkl`/`mtl-h`; the LSC API (`memory.md`
  §6: blocks, cache hints, stateless blocks, gathers, atomics);
  `cm_spec_constant` (`language.md` §8); the run-time `-cmc` source path
  (§5); the target macros (`targets.md`); every example.
- **Compile errors on DG2:** LSC 2D blocks (`lsc/block2d.h`) need PVC or
  Xe2.
- **Not tried:** `-vc-use-bindless-buffers/images`.

### Checking the versions in play

```bash
dpkg-query -W -f='${Package} ${Version}\n' intel-opencl-icd libze-intel-gpu1 libze1 \
  libigc2 libigdfcl2 intel-ocloc intel-igc-cm libigdgmm12 'intel-media-va-driver*' libigfxcmrt7
apt-cache policy intel-opencl-icd libigc2 intel-igc-cm   # which repository each comes from
ocloc --version                                          # compute-runtime (NEO) of the system ocloc
readlink -f /usr/lib/x86_64-linux-gnu/libigc.so.2        # the IGC actually loaded
clinfo | grep -E 'Device Name|Driver Version'            # runtime version per GPU (= NEO)
sycl-ls --verbose                                        # the same through oneAPI (L0 and OpenCL)
cmc --version | grep -a 'clang version'                  # 11 = intel-igc-cm 1.0.225 (8 = the 2021 SDK)
lspci -nnk | grep -A3 -E 'VGA|Display'                   # PCI id and kernel driver (i915 / xe)
ls /usr/local/lib | grep -E 'igc|igdfcl|opencl-clang'    # leftovers of manual .deb/tarball installs
```

Then compare the versions with the newest release on
https://dgpu-docs.intel.com (release notes → package list for your OS)
and with the release pages of `intel/compute-runtime`, which name the IGC
each NEO release was tested with.
- Things to look for: no `intel-igc-cm` at all, an IGC from another
  release than the one `intel-igc-cm` belongs to, and stale manual IGC
  installs in `/usr/local/lib` (NEO 26.x loads `libigc.so.2`, so an old
  IGC 1.x there is unused but confusing).
- `cm_ocl.hpp` reports the GPU IP version (e.g. 12.55.8 for DG2-G10,
  12.1.0 for Rocket Lake), which is what `-march`/`-device` must match.
  Note that `rkl` is not `tgllp`: a `tgllp` binary is rejected on Rocket
  Lake (tested on a UHD 750).
