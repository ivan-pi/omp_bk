// target_genx.cpp -- specializing a CM kernel for the GPU it is compiled for.
//
// The target is fixed at compile time (cmc -march=<alias|ip> for a current
// cmc, -mcpu=<NAME> for the 2021 SDK), and the compiler describes it with
// predefined macros:
//   __CM_INTEL_TARGET_MAJOR/MINOR/REVISION  IP version, e.g. 12.55.8 = DG2 (current cmc only)
//   __CM_INTEL_TARGET(maj, min, rev)        encodes a triple like the OpenCL
//   __CM_INTEL_TARGET_ID                    CL_DEVICE_IP_VERSION_INTEL word
//   CM_GENX                                 legacy number (1200 TGLLP, 1201 RKL,
//                                           1271 DG2, 1280 PVC, 1290 BMG, ...);
//                                           deprecated upstream, not monotonic in age
//   CM_GEN12, CM_XEHPG, CM_XE2_HPG, ...     legacy platform macros
//   CM_HAS_LSC, CM_HAS_DPAS, CM_HAS_LSC_UNTYPED_2D, CM_HAS_BF16, ...  features
// Prefer feature macros (CM_HAS_*), then IP ranges, over CM_GENX values.
// Full table: cm-compiler clang/compute_sdk/docs/cmcuserguide/platforms.rst.
//
//   target_info  writes the macros it was compiled with, for the host to check
//   copy_best    LSC 256 B blocks where CM_HAS_LSC, legacy 128 B OWord blocks otherwise
#include <cm/cm.h>

// Portable spelling for both compilers: the 2021 SDK has no IP macros.
#if defined(__CM_INTEL_TARGET_MAJOR)
#define TGT_MAJOR __CM_INTEL_TARGET_MAJOR
#define TGT_MINOR __CM_INTEL_TARGET_MINOR
#define TGT_REV __CM_INTEL_TARGET_REVISION
#define TGT_AT_LEAST(maj, min) (__CM_INTEL_TARGET_ID >= __CM_INTEL_TARGET(maj, min, 0))
#else
#define TGT_MAJOR 0
#define TGT_MINOR 0
#define TGT_REV 0
#define TGT_AT_LEAST(maj, min) 0
#endif

#if defined(CM_HAS_LSC)
constexpr int HAS_LSC = 1;
#else
constexpr int HAS_LSC = 0;
#endif
#if defined(CM_HAS_DPAS)
constexpr int HAS_DPAS = 1;
#else
constexpr int HAS_DPAS = 0;
#endif

// An IP-range check, e.g. Xe-HPG (12.55) or newer, as one would gate a code path.
#if TGT_AT_LEAST(12, 55)
constexpr int XEHPG_OR_NEWER = 1;
#else
constexpr int XEHPG_OR_NEWER = 0;
#endif

extern "C" _GENX_MAIN_ void target_info(SurfaceIndex out [[type("buffer_t")]]) {
  vector<int, 8> v;
  v(0) = CM_GENX;
  v(1) = TGT_MAJOR;
  v(2) = TGT_MINOR;
  v(3) = TGT_REV;
  v(4) = HAS_LSC;
  v(5) = HAS_DPAS;
  v(6) = XEHPG_OR_NEWER;
  v(7) = __cplusplus / 100;
  write(out, 0, v);
}

constexpr int CHUNK = 256; // floats per thread

extern "C" _GENX_MAIN_ void copy_best(SurfaceIndex x [[type("buffer_t")]],
                                      SurfaceIndex y [[type("buffer_t")]]) {
  uint off = (cm_group_id(0) * cm_local_size(0) + cm_local_id(0)) * CHUNK * 4;
  vector<float, CHUNK> v;
#if defined(CM_HAS_LSC)
  constexpr int B = 64; // LSC transposed block: up to 64 DWords = 256 B
#pragma unroll
  for (int i = 0; i < CHUNK / B; ++i) {
    v.select<B, 1>(i * B) = cm_load<float, B>(x, off + i * B * 4);
  }
#pragma unroll
  for (int i = 0; i < CHUNK / B; ++i) {
    cm_store<float, B>(y, off + i * B * 4, v.select<B, 1>(i * B));
  }
#else
  constexpr int B = 32; // legacy OWord block: up to 128 B
#pragma unroll
  for (int i = 0; i < CHUNK / B; ++i) {
    read(x, off + i * B * 4, v.select<B, 1>(i * B));
  }
#pragma unroll
  for (int i = 0; i < CHUNK / B; ++i) {
    write(y, off + i * B * 4, v.select<B, 1>(i * B));
  }
#endif
}
