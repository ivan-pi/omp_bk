# Reading the generated code: vISA, Gen assembly, IGC dumps and flags

Use this when a CM kernel is slower than expected and you need to see what
the compiler did. Everything below was checked on DG2 (an Arc A770) with
ocloc 26.40 / IGC 2.42, using dumps of `examples/spmv_genx.cpp`,
`dpas_genx.cpp` and saxpy.

## Contents
1. Sources
2. Getting the dumps (tested flags)
3. Reading vISA (`*.visaasm`)
4. Reading Gen assembly (`*.asm`)
5. A performance checklist from the assembly

---

## 1. Sources

- **vISA specification** (the virtual ISA that CM and ESIMD compile to, and
  the input of the finalizer):
  https://github.com/intel/intel-graphics-compiler/tree/master/documentation/visa
  - `3_execution_model.md`: SIMD execution, masks.
  - `5_operands.md`: regions and operand kinds.
  - `6_instructions.md` and `instructions/`: one file per instruction.
  - `8_appendix_visa_assembly_syntax.md`: `.decl`, `.input` and
    `.kernel_attr`.
  - `appendix_instruction_by_platform.md`: which instructions exist where.
  vISA "provides a portable ISA that spans multiple GEN platforms", and most
  vISA instructions map 1:1 to a Gen instruction. The differences are that
  vISA uses variables instead of registers and has variable-length
  instructions.
- **"Introduction to GEN Assembly"** (Intel, R. Ioffe, 2019; Gen8/Gen9 and
  OpenCL-centred, but the region and instruction syntax still applies):
  https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-gen-assembly.html
- **IGC shader dumps:**
  https://github.com/intel/intel-graphics-compiler/blob/master/documentation/shader_dumps_instruction.md
- **IGC configuration flags** (`IGC_<flag>=<value>` environment variables;
  the "Release build" column says whether a flag works in a normal driver
  build): https://github.com/intel/intel-graphics-compiler/blob/master/documentation/configuration_flags.md

## 2. Getting the dumps

Any path that runs IGC dumps, whether `ocloc`, `clBuildProgram`,
`zeModuleCreate` or the SYCL kernel compiler. For CM (the "VC" backend) the
files are named `VC_<hash>_*`.

| Flag (environment variable) | Effect on the CM/VC path (tested) |
|---|---|
| `IGC_ShaderDumpEnable=1` | **works**: writes `after_spirv_reader.ll`, `after_ir_adaptors.ll`, `optimized.ll`, `final.ll`, `<kernel>.visaasm`, `<kernel>.asm`, `.zeinfo`, options. Default directory `/tmp/IntelIGC/<app>_<pid>/` |
| `IGC_DumpToCurrentDir=1` | **works**: dump into the current directory |
| `IGC_DumpToCustomDir=<dir>` | **works**: the parent directory must exist |
| `IGC_ShaderDumpPidDisable=1` | **works**: no PID in the directory name, so repeated runs overwrite (handy for diffs) |
| `IGC_ShaderDumpRegexFilter='.*\.(asm\|visaasm)$'` | **works**: keep only matching files (here only the asm and vISA, plus `.zeinfo`/`.dat`) |
| `IGC_ShaderDumpEnableAll=1` | dumps the LLVM IR after every pass (large) |
| `IGC_ShaderDumpEnableRAMetadata=1` | produced nothing extra on the VC path |
| `IGC_DumpVISAASMToConsole=1` | printed nothing on the VC path |
| `IGC_VISAOptions="..."` / `-Xfinalizer "..."` | accepted; `-printregusage` printed nothing, so read the `.asm` header instead |
| `IGC_ShaderOverride=1` | replaces `.visaasm`/`.asm` with edited copies from `/tmp/IntelIGC/ShaderOverride` (same hash in the name) to try hand changes. Marked Diagnostic and may need a debug build |
| `IGC_ShaderPassDisable`, `IGC_ShaderDisplayAllPassesNames` | debug IGC builds only |

`scripts/cm_build.sh -k` sets `ShaderDumpEnable` and `DumpToCurrentDir` and
collects everything in `igc_dump/`. Typical comparison loop:
```bash
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=$PWD/A IGC_ShaderDumpPidDisable=1 \
IGC_ShaderDumpRegexFilter='.*\.(asm|visaasm)$' ocloc compile -q -file k.spv -spirv_input \
  -device dg2 -options "-vc-codegen" -output k -output_no_suffix
# change the kernel or the options, dump to B, then diff A/*.asm B/*.asm
```

## 3. Reading vISA (`*.visaasm`)

```
.kernel "spmv_csr"
.decl V39 v_type=G type=b num_elts=32 align=GRF        // general variable: 32 bytes, GRF-aligned
.decl V40 v_type=G type=d num_elts=1 alias=<V37, 0>    // alias = a view (format/select of another variable)
.decl P1 v_type=P num_elts=1                           // predicate (flag) variable
.decl T6 v_type=T num_elts=1                           // surface (SurfaceIndex)
.input T6 offset=88 size=4                             // kernel argument in the payload
.implicit_LOCAL_ID V32 offset=32 size=6
.kernel_attr NumGRF=128     .kernel_attr SimdSize=32     .kernel_attr Target="cm"
    lsc_load.ugm (M1, 2)  V40:d32  bti(0x1)[V104]:a32     // 2-lane gather of rowptr
    cmp.lt (M1, 1) P1 V107(0,0)<0;1,0> 0x9:ud            // scalar compare: len < 9
    (P1) jmp (M1, 1) BB_1                                 // uniform branch (scalar if)
```
- `v_type` is G (general), A (address), P (predicate), S (sampler) or T
  (surface). Every CM `vector`/`matrix` becomes a G variable; a
  `select`/`format` becomes an `alias` or a region operand, not a copy.
- Operands are written `V(row,col)<vstride;width,hstride>`. These are CM's
  `select<>` regions, still visible here before register allocation.
- `(M1, 16)` is the execution mask group and size. `SimdSize=32` on a CM
  kernel is the dispatch mask width, not one width per instruction:
  every instruction has its own size.
- The memory messages appear by name (`lsc_load.ugm`, `oword_ld`, `gather`,
  `dword_atomic`...), so this is the place to check which message a CM read
  became.

## 4. Reading Gen assembly (`*.asm`)

Header and footer lines (start with `//`):
```
//.platform DG2
//.thread_config numGRF=128, numAcc=4, numSWSB=16      <- register file size (256 with -double-GRF)
//.instCount 39                                         <- static instruction count
//.numALUInst: 26 ...                                   <- footer: ALU vs send mix, sync counts
```
A GRF spill shows up as extra `send`s to scratch memory (instruction lines,
not `//` comments, mentioning `scratch`/spill) and as a larger instruction
count. **No spill or scratch traffic is the first thing to confirm.**
`//.declare %scratchloc` is always present and means nothing.
`//.spill flag store N` / `//.spill flag load N` count **flag-register**
spills: masks wider than the few flag registers (for example 64- or 128-lane
compare results) are moved to and from GRFs. They cost instructions but no
memory traffic. Narrower lanes or fewer live masks reduce them, which matters
only when the kernel is ALU-bound (`examples/argmax_genx.cpp` has 51-105 of
them and still runs at full bandwidth).

Instruction form: `[(pred)] op (exec_size|M_offset) dst src0 [src1 [src2]] {swsb}`
```
(W)     mad (1|M0)        r127.2<1>:ud  r127.2<0;0>:ud  r127.0<0;0>:uw  0x20:uw   {I@1}
        load.ugm.d32x32t.a32 (1|M0)  r11:4  bti[2][r10:1]   {A@2,$3}   // LSC block load: 32 dwords -> r11..r14
(W)     send.dc0 (8|M0)   r1  r127  null:0  0x0  0x021842FD  {A@1,$0}  // legacy data-port message (oword block read)
        cmp (16|M16) (lt)f1.0  null<1>:d  r21.0<1;1,0>:ud  r5.1<0;1,0>:ud {I@2}
(f0.0.any32h) goto (32|M0)  BB_2  BB_2                                 // SIMD control flow / uniform branch
        dpas.8x8 (8|M0)   r17:f  r17:f  r9:hf  r1.0:hf  {$0}            // XMX: systolic depth 8, repeat 8
```
- **Registers:** `r0`–`r127` (or `r255`) are 32 bytes each on Gen9–XeHPG
  and 64 bytes on XeHPC/Xe2+. `r0` holds the thread payload (group ids);
  the top register is reserved for the end-of-thread message. `f0.0`/`f1.0`
  are flag registers, `acc0` the accumulator, `a0` the address register
  (indirect operands), `null` the discard destination.
- **Regions** `rN.sub<V;W,H>:type`: start at sub-element `sub` of register N,
  take W elements H apart per row, rows V apart. `<1;1,0>` or `<1>` is
  contiguous, `<0;1,0>` is a scalar broadcast, and `<16;8,2>` reads every
  second element. Your CM `select<size,stride>` should appear directly as
  such a source region. If it shows up as a separate `mov` first, it did not
  fold.
- **Types:** `:b/:ub` (8-bit), `:w/:uw` (16), `:d/:ud` (32), `:q/:uq` (64),
  `:hf` (half), `:bf` (bfloat16), `:f` (float), `:df` (double).
- **`(N|Mk)`** is the execution size and the channel-mask offset. A SIMD32
  CM statement on 32-bit data usually becomes two SIMD16 instructions,
  `(16|M0)` and `(16|M16)`. `(W)` means NoMask: it runs regardless of the
  execution mask, as for scalar bookkeeping.
- **`{...}` is software scoreboarding (Gen12+):** `$n` is a token set by a
  `send`/`dpas`, `$n.dst`/`$n.src` waits on it, `A@1`/`I@2` waits on the
  previous ALU or integer instruction, and `sync.nop` is an explicit wait.
  Many waits right after `send`s mean the code is latency-bound.
- **Memory:** `load.ugm`/`store.ugm` are LSC messages (DG2+). `.d32x32t` is
  32 dwords in one transposed block, `.d32` a per-lane gather, and `bti[k]`
  the surface. `send.dc0`/`send.dc1` are legacy data-port messages; the
  `// comment` decodes them (e.g. "oword aligned block read x4").
  `send.gtwy ... {EOT}` ends the thread.
- **Fixed sends every thread pays (DG2, tested on saxpy):** two `send.dc0`
  OWord block reads at the start that load the thread payload (kernel
  arguments; DG2 keeps it in memory, `thread_payload_in_memory`); a
  `send.ugm` "fence ... scoped to tile" plus `mov` commit before the end
  when the kernel stored to memory; and the `send.gtwy {EOT}`. So 64 floats
  of saxpy issue 6 data sends + 4 fixed ones. The data width doesn't change
  them: amortize them by giving each thread more work. The SIMT-entry
  attribute `_SIMT16_` (`__declspec(genx_SIMT(16))`) is not a way out: on a
  kernel it crashed both IGC 2.42 (ocloc segfault) and IGC 2.11 ("Internal
  Compiler Error").
- **Not available on newer GPUs:** `dp4`/`dph`/`line`/`pln` disappear on
  Gen12+, where `cm_dp4` became `mul`/`mad` (`examples/spmv_genx.cpp`).

## 5. A performance checklist from the assembly

1. `numGRF` and spills: no scratch traffic. If there is some, shrink the
   per-thread working set or try `-double-GRF`.
2. Count the memory `send`s/`load`s per thread and compare with the minimum
   your data needs (bytes ÷ the largest message). Extra sends usually mean
   misaligned or under-sized reads.
3. Look for `mov`s that only rearrange data. A `select` that should have
   been a source region, or a variable select offset (indirect `a0`
   addressing), costs real instructions. Unroll so the offsets become
   constants, and bind offsets computed by `constexpr` functions to
   `constexpr` variables (`references/language.md` §8): a `call`/`ret`
   pair in the asm means a helper was not inlined.
4. Check the arithmetic width: `(16|M0)` FP32 `mad`s are full width on DG2.
   Many `(1|M0)` instructions in the hot loop mean scalar code that should
   be vectors.
5. Look at the scoreboard: long chains of `$n.dst` waits after loads mean
   too little independent work between a load and its use. Issue the loads
   earlier or software-pipeline them.
6. Compare two variants with the regex-filtered dumps and `diff`.
