# Target macros: specializing CM code for a GPU

## Contents
1. Where the codes come from
2. Finding the IP version of your GPU
3. Macros the compiler defines
4. Platform table
5. Idioms
6. Fat binaries: one file for several GPUs
7. Pitfalls
8. Tested example

---

## 1. Where the codes come from

A CM kernel is compiled for **one** target, and the compiler describes it
with predefined macros. The authoritative tables are in the cm-compiler
repository (https://github.com/intel/cm-compiler):

| File | Contents |
|---|---|
| `clang/compute_sdk/docs/cmcuserguide/platforms.rst` | **the table:** IP version, `-march` alias, device names, legacy platform macro, `CM_GENX` (both legacy columns marked deprecated) |
| `clang/compute_sdk/docs/cmcuserguide/cmcuserguide.rst` | `-march`, the `__CM_INTEL_TARGET_*` macros, "the triple uniquely identifies the platform" |
| `clang/lib/Headers/cm/include/cm/cm_target.h` | IP version → `CM_GENX` and the legacy `CM_GEN12`/`CM_XEHPG`/... macros, and `__CM_INTEL_TARGET(maj,min,rev)` |
| `clang/lib/Headers/cm/include/cm/cm_has_instr.h` | feature macros `CM_HAS_*`, with the platform condition of each |
| `clang/lib/Driver/ToolChains/Arch/GenXPlatforms.cpp` | `-march` alias → IP version (`{"dg2", encodeGmdId(12, 55, 8)}`, ...) |
| `clang/lib/Basic/Targets/GenX.cpp` | where the target macros get defined |

The installed copies of the two headers are the quickest reference:
`/usr/include/cm/` from `intel-igc-cm` (`~/opt/intel-cm-<build>/root/usr/include/cm/`
for the user-space install). Upstream adds new
GPUs (NVL, CRI, ...) that a given packaged cmc may not know yet.

## 2. Finding the IP version of your GPU

```bash
sycl-ls                         # e.g. "... Intel(R) Arc(TM) A770 Graphics 12.55.8 [...]"
ocloc ids dg2                   # alias -> IP: 12.55.8  (ocloc ids rkl -> 12.1.0)
lspci -nn | grep -E 'VGA|Display'   # PCI id, e.g. [8086:56a0]
ocloc compile -file k.cl -device 0x56a0 ...   # ocloc also takes a PCI id: "Auto-detected ... dg2-g10-a0"
cmc -march=dg2 -E -dM -o - k.cpp | grep -E 'CM_GENX |__CM_INTEL_TARGET_|CM_HAS_'   # what a target defines
```

At run time: OpenCL `CL_DEVICE_IP_VERSION_INTEL` (0x4250, `cmocl::ip_version`)
or Level Zero `ze_device_ip_version_ext_t`. The raw word uses the same
encoding as `__CM_INTEL_TARGET(maj, min, rev)`, namely
`major << 22 | minor << 14 | revision` (e.g. DG2-G10: 0x30dc008 = 12.55.8).
**clinfo decodes it wrongly**: "Device IP (Intel) 0x30dc008 (12.220.8)"
and "0x3004000 (12.4.0)" are really 12.55.8 and 12.1.0.

## 3. Macros the compiler defines

| Macro | Meaning (`cmc -march=<device>`) |
|---|---|
| `__CM_INTEL_TARGET_MAJOR/MINOR/REVISION` | IP triple, e.g. 12 / 55 / 8 |
| `__CM_INTEL_TARGET(maj,min,rev)`, `__CM_INTEL_TARGET_ID` | encoder + the target's word, usable in `#if` |
| `CM_GENX`, `CM_GENX_REVID` | derived from the IP in `cm_target.h`; deprecated |
| `CM_GEN9`, `CM_GEN11`, `CM_GEN12`, `CM_XEHP`, `CM_XEHPG`, `CM_XEHPC`, `CM_XELPG`, `CM_XELPGPLUS`, `CM_XE2_HPG`, `CM_XE2_LPG`, `CM_XE3_LPG` | legacy platform macros |
| `CM_HAS_LSC`, `CM_HAS_DPAS`, `CM_HAS_LSC_UNTYPED_2D`, `CM_HAS_LSC_TYPED_2D`, `CM_HAS_BF16`, `CM_HAS_TF32`, `CM_HAS_DP4A`, `CM_HAS_BFN`, `CM_HAS_BIT_ROTATE`, `CM_HAS_IEEE_DIV_SQRT`, ... | defined only when supported |
| `CM_HAS_<X>_CONTROL` | `static_assert` used inside the headers: an unsupported intrinsic is a compile error naming the feature |

The 2021 SDK cmc defines only `CM_GENX` and part of the `CM_HAS_*` set, from
its `-mcpu` (`legacy-sdk.md`).

Feature conditions in `cm_has_instr.h` (current): LSC `CM_GENX >= 1271`
(DG2+); untyped 2D LSC `>= 1280` (PVC, Xe2, Xe3); typed LSC / 2D
`__CM_INTEL_TARGET_MAJOR >= 20`; `CM_HAS_SAMPLE_UNORM` `< 1270`; legacy
typed atomics majors 9..19. `CM_HAS_DPAS` etc. come from the compiler.

## 4. Platform table

From `platforms.rst` (one row per platform; revisions omitted) and checked
with `cmc -march=<alias> -E -dM` for the packaged 1.0.225 cmc.

| GPU | IP version | `-march` alias (also `-march=<ip>`) | Legacy macro | `CM_GENX` | LSC | DPAS | 2D LSC |
|---|---|---|---|---|---|---|---|
| Skylake | 9.0.x | `skl` | CM_GEN9 | 900 | | | |
| Broxton/Apollo Lake | 9.3 | | CM_GEN9 | 920 | | | |
| Kaby/Coffee/Comet Lake | 9.1-9.7 | | CM_GEN9_5 | 950 | | | |
| Gemini Lake | 9.4 | | CM_GEN9_5 | 970 | | | |
| Ice Lake | 11.0 | `icllp` | CM_GEN11 | 1150 | | | |
| Tiger Lake | 12.0.0 | `tgllp` | CM_GEN12 | 1200 | | | |
| Rocket Lake (UHD 750) | 12.1.0 | `rkl` | CM_GEN12 | 1201 | | | |
| Alder/Raptor Lake-S | 12.2.0 | `adls` (`adl-s`) | CM_GEN12 | 1230 | | | |
| Alder/Raptor Lake-P | 12.3.0 | `adlp` (`adl-p`) | CM_GEN12 | 1220 | | | |
| Alder Lake-N | 12.4.0 | `adln` | CM_GEN12 | 1240 | | | |
| DG1 | 12.10.0 | `dg1` | CM_GEN12 | 1210 | | | |
| Xe-HP SDV | 12.50.4 | `xehp-sdv` | CM_XEHP | 1270 | | yes | |
| Arc A770/A750, Flex 170 (DG2-G10) | 12.55.8 | `dg2`, `acm-g10` | CM_XEHPG | 1271 | yes | yes | |
| Arc A380 (DG2-G11) | 12.56.5 | `acm-g11` | CM_XEHPG | 1271 | yes | yes | |
| DG2-G12 | 12.57.0 | `acm-g12` | CM_XEHPG | 1271 | yes | yes | |
| Ponte Vecchio (Max) | 12.60.7 | `pvc` | CM_XEHPC | 1280 | yes | yes | yes |
| Meteor Lake-U (12.70.4), Meteor Lake-H e.g. Core Ultra 7 155H (12.71.4), Arrow Lake-S/U | 12.70 / 12.71 | `mtl-u` (= `mtl`), `mtl-h` | CM_XELPG | 1275 | yes | **no** | |
| Arrow Lake-H | 12.74.4 | `arl-h` | CM_XELPGPLUS | 1276 | yes | yes | |
| Battlemage (B580, G21) | 20.1.0 | `bmg` | CM_XE2_HPG | 1290 | yes | yes | yes |
| Lunar Lake | 20.4.4 | `lnl` | CM_XE2_LPG | 1295 | yes | yes | yes |
| Panther Lake | 30.0.4 | `ptl` | CM_XE3_LPG | 1300 | yes | yes | yes |
| Nova Lake, Crescent Island | 30.x / 35.x | upstream only | CM_XE3_LPG / CM_XE3P_* | 1300 / 1360 / 1380 | | | | — |

The 2021 SDK cmc's `-mcpu` names and its workaround for newer GPUs are in
`legacy-sdk.md`.

## 5. Idioms

```cpp
// 1. Feature first: the most robust test, also correct for future GPUs.
#if defined(CM_HAS_LSC)
  v = cm_load<float, 64>(buf, off);          // 256 B LSC block
#else
  read(buf, off, v.select<32, 1>(0)); ...    // 128 B OWord blocks
#endif

// 2. IP range (current cmc): "Xe-HPG or newer", "this exact GPU family".
#if defined(__CM_INTEL_TARGET_ID) && __CM_INTEL_TARGET_ID >= __CM_INTEL_TARGET(12, 55, 0)
  ...
#endif
#if __CM_INTEL_TARGET_MAJOR == 12 && __CM_INTEL_TARGET_MINOR == 1   // RKL only
  ...
#endif

// 3. Code that must also build with the 2021 cmc (legacy-sdk.md), which has only CM_GENX.
#if defined(__CM_INTEL_TARGET_MAJOR)
#define TGT_AT_LEAST(maj, min) (__CM_INTEL_TARGET_ID >= __CM_INTEL_TARGET(maj, min, 0))
#else
#define TGT_AT_LEAST(maj, min) 0                  // or a CM_GENX test
#endif

// 4. Tuning constants per target, in one place:
#if defined(CM_HAS_LSC)
constexpr int BLOCK = 64;
#else
constexpr int BLOCK = 32;
#endif
```

For one binary that serves several GPUs, see §6.

## 6. Fat binaries: one file for several GPUs

Source: `ocloc compile --help` (`-device`, `-device_options`, `-exclude_ir`)
and `ocloc --help` (`concat`, `ids`), from Intel's compute runtime
(https://github.com/intel/compute-runtime). Everything below was tested on
an Arc A770 (DG2) and a UHD 750 (Gen12LP, RKL) with `target_genx.cpp` and
`patterns_genx.cpp`.

Use the `ocloc` of the cmc's own release (the system one after a
system-wide install, pipeline.md §7). With the user-space install that is
`ocloc-cm`, **not an Intel tool** but this skill's wrapper
(`scripts/install_cm_toolchain.sh`); the commands below were run with it.

```bash
# From CM source: the CM front end runs once PER DEVICE, so each GPU gets
# its own specialization (CM_HAS_LSC, __CM_INTEL_TARGET_*, ...)
ocloc compile -file k_genx.cpp -device dg2,rkl -options "-cmc" -output k -output_no_suffix

# From one SPIR-V: one specialization (the cmc -march target), finalized per device
ocloc compile -file k.spv -spirv_input -device dg2,rkl -options "-vc-codegen" \
         -device_options dg2 "-ftranslate-legacy-memory-intrinsics" -output k -output_no_suffix

# Device lists, ranges, IP versions, PCI ids
-device dg2,rkl            # two targets
-device 12.55.8,12.1.0     # the same by IP version
-device rkl:acm-g12        # every target in the range (13 with ocloc 25.18: RKL, ADL-S/P/N, DG1, all DG2 steppings)
-device dg2:  /  -device :rkl   # all newer / all older than (inclusive)
-device 0x56a0             # PCI id -> "Auto-detected target ... dg2-g10-a0"

ar t k                     # members: 64.dg2, 64.12.1.0, ... (+ generic_ir for SPIR-V input)
ocloc concat a b -out ab.ar   # merge fat binaries
```

- **Loading:** the output (no suffix with `-output_no_suffix`) is an `ar`
  archive. Load it like any binary (`clCreateProgramWithBinary`;
  `cmocl::load_program` with a `.bin` name), and the driver picks the
  member matching the device. One `dg2,rkl` file passed on both
  DG2 and RKL, and a 26.40 driver loads archives built by ocloc 25.18.
- **`generic_ir`:** a fat binary built from SPIR-V also embeds the SPIR-V.
  The driver JIT-compiles it for a GPU not in the list; for a current
  cmc's SPIR-V that needs the matched IGC. `-exclude_ir` drops it. CM
  source builds (`-cmc`) contain native members only.
- **Per-device backend options:** `-device_options <device> "<opts>"`
  works for SPIR-V input, e.g. the DG2-only
  `-ftranslate-legacy-memory-intrinsics`, which RKL rejects. With `-cmc`,
  `-device_options` makes the build fail, because those options go to the
  front end. A `-cmc` build of `patterns_genx.cpp` (SIMD16 legacy atomics)
  needs no flag and passed on DG2 and RKL.
- **Size:** about 13 KB per member for `target_genx.cpp`; the 13-target
  range build was 170 KB.
- **Which to use:** build from CM source when the code specializes per
  GPU (`#if defined(CM_HAS_LSC)` etc.); from SPIR-V when the code is
  target-neutral. One binary per target, chosen at run time by
  `cmocl::ip_version(dev)`, is the alternative to a fat binary.

## 7. Pitfalls

- **`CM_GENX` doesn't follow the hardware's age:** ADL-P is 1220 and ADL-S
  1230; MTL (1275) sits between DG2 (1271) and PVC (1280) yet has no DPAS.
  So `CM_GENX >= 1271` doesn't mean "has DPAS". Test `CM_HAS_DPAS`, or use
  IP ranges. Upstream deprecates `CM_GENX` and the platform macros.
- **The macros describe the compile target, not the GPU.** SPIR-V carries
  its specialization along: a `-march=rkl` SPIR-V finalized for DG2
  runs, but uses the RKL code path. `host_target.cpp` detects this.
- **Revisions:** stepping differences (12.55.0 ... 12.55.8) rarely matter
  for CM code, so compare major.minor unless a stepping erratum applies.
- **`-march` takes aliases and raw IPs** (`-march=12.55.8`). Names the
  packaged cmc doesn't know yet (NVL, CRI) fail; the raw IP may fail too.

## 8. Tested example

`examples/target_genx.cpp` + `host_target.cpp`: `target_info` writes
`CM_GENX`, the IP triple, `CM_HAS_LSC`/`CM_HAS_DPAS` and an IP-range test,
and the host compares them with the device's IP; `copy_best` uses LSC
blocks where `CM_HAS_LSC` is defined and OWord blocks otherwise.

Tested on an Arc A770 (DG2) and a UHD 750 (RKL):

| Build | GPU | Reported | Result |
|---|---|---|---|
| current cmc `-march=dg2` (.spv under `cm_run`, .bin) | A770 12.55.8 | CM_GENX 1271, IP 12.55.8, LSC 1, DPAS 1 | PASS, LSC path |
| current cmc `-march=rkl` (.bin) | UHD 750 12.1.0 | CM_GENX 1201, IP 12.1.0, LSC 0 | PASS, legacy path |
| current cmc `-march=rkl` (.spv) | A770 | IP 12.1.0 ≠ 12.55 | runs, but the target check FAILs as intended |
