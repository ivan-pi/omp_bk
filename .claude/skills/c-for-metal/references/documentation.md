# CM documentation sources and how they differ

## Learning resources (full list)

Recommend these when someone wants to learn CM:

- **Language specification** (the primary reference), in the compiler
  repository: https://github.com/intel/cm-compiler, at
  `clang/compute_sdk/docs/cmlangspec/cmlangspec.rst`. The same folder has
  the compiler user guide (`cmcuserguide/`), a 15-part tutorial
  (`cmtutorial/`), a porting guide and the OpenCL/Level Zero runtime notes
  (`cmoclrt/`). The spec's example sources are in `clang/test/CMFE/cmlangspec/`.
  An older HTML copy came with the 2021 SDK (`legacy-sdk.md` §5).
  The comparison of the doc copies is below.
- **Paper:** G.-Y. Lueh, K. Chen, G. Chen, J. Fuentes, W.-Y. Chen, F. Fu,
  H. Jiang, H. Li, D. Rhee, "C-for-Metal: High Performance SIMD Programming
  on Intel GPUs", CGO 2021, pp. 289-300,
  https://doi.org/10.1109/CGO51591.2021.9370324 (preprint
  https://arxiv.org/abs/2101.11049). It covers the motivation, the
  language, the compiler (rdregion/wrregion, baling) and the benchmarks.
  Its artifact repository is https://github.com/jfuentes/C-for-Metal_CGO2021.
- **Talk:** the CGO presentation of the paper,
  https://www.youtube.com/watch?v=jOBUmgwJbow
- **Intel article:** "C for Metal - The Precious Metal for Computing on
  Intel Graphics Technology",
  https://www.intel.com/content/www/us/en/developer/articles/technical/intel-c-for-metal-the-precious-metal-for-computing-on-intel-graphics-cards.html
- **CPU emulator** (archived): https://github.com/intel/cm-cpu-emulation
- **Target/platform codes** (IP version, `-march` alias, `CM_GENX`):
  `clang/compute_sdk/docs/cmcuserguide/platforms.rst` in the cm-compiler
  repo, plus `cm_target.h` / `cm_has_instr.h` in its headers. A summary,
  with ways to find a GPU's IP, is in `targets.md`.
- **Prebuilt current compiler and driver versions:** https://dgpu-docs.intel.com
  (Intel's GPU driver documentation; the release notes have per-release
  package lists, e.g. `intel-igc-cm_1.0.225.54093` in LTS 2523.40). Use it
  to find the `intel-igc-cm` release that matches a driver, and to compare
  the installed driver with the latest. The paths change often, so start
  from the root. The packages come from `repositories.intel.com/gpu/ubuntu`;
  the install and version-check commands are in `pipeline.md` §7.
- **C for Metal Development Package, development reference** (Intel page
  dated 2019-02-22):
  https://www.intel.com/content/www/us/en/content-details/738812/c-for-metal-development-package-development-reference.html
  (download: https://cdrdv2.intel.com/v1/dl/getContent/738812?fileName=c-for-metal-development-package-reference.zip).
  Intel links move or disappear often; if these are dead, search
  intel.com for the title. Checked on 2026-11-16: it is a 73 KB zip with
  **only the 15-part tutorial**, from 2019 and still titled "CM (C for
  Media) Tutorial", with no spec and no compiler. The content matches
  GitHub's `cmtutorial/`, but the zip's code excerpts are correct where
  several of GitHub's `literalinclude` line ranges have drifted. Both teach
  the legacy CM-runtime host API. Details are in `references/documentation.md`.
- **Work citing the CM paper** (context, not tutorials):
  - D. Mustafa et al., "MIMD Programs Execution Support on SIMD Machines: A
    Holistic Survey", IEEE Access 12 (2024), doi:10.1109/ACCESS.2024.3372990.
    It lists CM under research programming models for SIMD machines: an
    explicit interface for SIMD size, register management and cross-lane
    sharing, harder to program but faster on mixed serial/parallel code
    with irregular memory access. It sits alongside ISPC.
  - D. Wang et al., "ML-Triton, A Multi-Level Compilation and Language
    Extension to Triton GPU Programming", arXiv:2503.14985 (2025). Triton
    for Intel GPUs lowers either to SIMT (GenISA intrinsics) or to SIMD via
    the **VectorCompute intrinsics**, the same `llvm.genx.*` layer CM
    produces. That backend is how CM's design lives on under ESIMD and
    Triton.
- **ESIMD learning material** (spec, examples README, the 122-file
  `sycl/test-e2e/ESIMD` suite with ports of the CM tutorial workloads, and
  the oneAPI GPU Optimization Guide chapter "Optimizing Explicit SIMD
  Kernels"): see `references/esimd.md`, "Where to learn ESIMD".
- **Successor in SYCL:** the ESIMD extension,
  https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd

## Comparison of the doc copies

There are three copies of the CM docs. They were compared on 2026-11-16;
which one to use depends on the question.

| Source | What it contains | Date | Use it for |
|---|---|---|---|
| GitHub `intel/cm-compiler`, `clang/compute_sdk/docs/` | spec (`cmlangspec/`), CM template library (`cmlangspec/cmtl.rst`), compiler guide (`cmcuserguide/`), OpenCL/L0 runtime notes (`cmoclrt/`), porting guide, 15-part tutorial (`cmtutorial/`) | maintained, edited in 2026 | **the reference**: current language, LSC, XMX, address spaces, `-march` targets |
| 2021 CM SDK, `docs/external/` (HTML) | spec, compiler guide, OpenCL runtime notes; **no tutorial** | 2021-10-26 | the 2021 cmc and the removed media/video intrinsics; compared with GitHub in `legacy-sdk.md` §5 |
| Intel "C for Metal Development Package - Development Reference" zip (page 738812) | **only the tutorial**: 16 HTML pages titled "CM (C for Media) Tutorial", "CM 6.0 documentation", © 2009-2016 | built 2019-02-22 | the tutorial with correct code excerpts, and history |

## Intel development-package zip

- Page: https://www.intel.com/content/www/us/en/content-details/738812/c-for-metal-development-package-development-reference.html
- Download link: https://cdrdv2.intel.com/v1/dl/getContent/738812?fileName=c-for-metal-development-package-reference.zip
  It redirects to a signed, expiring CloudFront URL. It's a 73 KB zip with
  `reference/cmtut.html` and `reference/cmtutorial01..15.html`; there is no
  compiler, spec or runtime in it. Intel links rot, so search for the
  title if this one is dead.
- Content: the same 15 tutorials as GitHub's `cmtutorial/`. The prose
  matches GitHub at 53-97% per chapter, and nearly all of the difference is
  Sphinx navigation and the rename "CM" → "C for Metal". In 2019, CM still
  stood for **"C for Media"**.
- The tutorial's host code uses the **legacy CM runtime** (`CreateCmDevice`,
  `CmDevice::CreateKernel`, `CmThreadSpace`, `CmQueue`, `CmBuffer::ReadSurface`,
  events), in both the zip and GitHub. It does not cover the OpenCL / Level
  Zero / SYCL path that this skill uses, and the hardware CM runtime
  (`libigfxcmrt`, Intel media driver) is often not installed and does not
  support DG2 or newer. Read the kernel
  side as current; translate the host side with SKILL.md "Host launch".
- **The zip's code excerpts are self-consistent; GitHub's have drifted.** On
  GitHub the snippets come from `.. literalinclude:: file :lines: a-b`, and
  some source files changed after the line ranges were written:
  - Chapter 10 (Event-Driven Synchronization): `:lines: 242-246` of
    `linear_up_walker1.cpp` now shows the execution-time query.
    `WaitForTaskFinished` has moved to line 230. Verified.
  - Chapter 3 includes `../../../test/open_examples/multi_kernels/multi_kernels.cpp`,
    which no longer exists in the repository (HTTP 404), so its multi-kernel
    host code is missing on GitHub.
  - The code similarity between the zip and GitHub is also low in chapters
    4, 6 and 8 (0.30-0.64; most others are 0.75-1.0), which points to more
    shifted excerpts.
  When a GitHub tutorial snippet doesn't match its prose, read the full
  source file in `cmtutorial/` or the zip's HTML.
