// lsc_genx.cpp -- the LSC (load/store cache) data-port API (cm_lsc.h),
// next to the legacy OWord block it replaces. DG2 (Xe-HPG) and newer only.
//
// Build with cmc -march=dg2 (or another LSC GPU). On pre-LSC targets (rkl, tgllp)
// every cm_load/cm_store/cm_atomic is a compile error (CM_HAS_LSC_CONTROL).
//
//   copy_oword   legacy read/write, 128 B per message (8 OWords)
//   copy_lsc     cm_load<float, 64> / cm_store<float, 64>: 256 B per message
//   copy_stream  the same with cache hints (L1 streaming / uncached store)
//   copy_ptr     flat-address cm_ptr_load / cm_ptr_store on SVM/USM pointers
//   gather_rev   SIMD16 LSC gather with per-lane byte offsets: y[i] = x[n-1-i]
//   hist_lsc     LSC atomics: cm_atomic<AtomicOp::IINC> on byte offsets
//
// Units: every LSC offset is in BYTES (block, gather and atomic alike), unlike
// the legacy API (scattered = elements, write_atomic = DWords).
// Block messages are "transposed": one address, 1..64 contiguous U32/U64
// elements (NElts in {1,2,3,4,8,16,32,64}); gathers are SIMD16 on DG2
// (SIMD32 from PVC on, details::lsc_default_simt()).
#include <cm/cm.h>

#ifndef LSC_CHUNK
#define LSC_CHUNK 512 // floats per thread
#endif
constexpr int CHUNK = LSC_CHUNK;
constexpr int OW = 32; // floats per legacy OWord block message (128 B)
constexpr int LB = 64; // floats per LSC block message (256 B)
static_assert(CHUNK % LB == 0, "a chunk must be whole LSC blocks");

_GENX_ inline uint thread_id() { return cm_group_id(0) * cm_local_size(0) + cm_local_id(0); }

extern "C" _GENX_MAIN_ void copy_oword(SurfaceIndex x [[type("buffer_t")]],
                                       SurfaceIndex y [[type("buffer_t")]]) {
  uint off = thread_id() * CHUNK * 4;
  vector<float, CHUNK> v;
#pragma unroll
  for (int i = 0; i < CHUNK / OW; ++i) {
    read(x, off + i * OW * 4, v.select<OW, 1>(i * OW));
  }
#pragma unroll
  for (int i = 0; i < CHUNK / OW; ++i) {
    write(y, off + i * OW * 4, v.select<OW, 1>(i * OW));
  }
}

extern "C" _GENX_MAIN_ void copy_lsc(SurfaceIndex x [[type("buffer_t")]],
                                     SurfaceIndex y [[type("buffer_t")]]) {
  uint off = thread_id() * CHUNK * 4;
  vector<float, CHUNK> v;
#pragma unroll
  for (int i = 0; i < CHUNK / LB; ++i) {
    v.select<LB, 1>(i * LB) = cm_load<float, LB>(x, off + i * LB * 4);
  }
#pragma unroll
  for (int i = 0; i < CHUNK / LB; ++i) {
    cm_store<float, LB>(y, off + i * LB * 4, v.select<LB, 1>(i * LB));
  }
}

// Cache hints are template arguments (L1, then L2); unsupported pairs are a
// compile-time error (lsc_check_cache_hint_load/store).
extern "C" _GENX_MAIN_ void copy_stream(SurfaceIndex x [[type("buffer_t")]],
                                        SurfaceIndex y [[type("buffer_t")]]) {
  uint off = thread_id() * CHUNK * 4;
  vector<float, CHUNK> v;
#pragma unroll
  for (int i = 0; i < CHUNK / LB; ++i) {
    v.select<LB, 1>(i * LB) = cm_load<float, LB, DataSize::Default, CacheHint::Streaming,
                                      CacheHint::Cached>(x, off + i * LB * 4);
  }
#pragma unroll
  for (int i = 0; i < CHUNK / LB; ++i) {
    cm_store<float, LB, DataSize::Default, CacheHint::Uncached, CacheHint::WriteBack>(
        y, off + i * LB * 4, v.select<LB, 1>(i * LB));
  }
}

extern "C" _GENX_MAIN_ void copy_ptr(svmptr_t x [[type("svmptr_t")]],
                                     svmptr_t y [[type("svmptr_t")]]) {
  const float *px = reinterpret_cast<const float *>(x);
  float *py = reinterpret_cast<float *>(y);
  uint off = thread_id() * CHUNK * 4;
  vector<float, CHUNK> v;
#pragma unroll
  for (int i = 0; i < CHUNK / LB; ++i) {
    v.select<LB, 1>(i * LB) = cm_ptr_load<float, LB>(px, off + i * LB * 4);
  }
#pragma unroll
  for (int i = 0; i < CHUNK / LB; ++i) {
    cm_ptr_store<float, LB>(py, off + i * LB * 4, v.select<LB, 1>(i * LB));
  }
}

// One thread reverses 64 floats: four SIMD16 gathers, one block store.
extern "C" _GENX_MAIN_ void gather_rev(SurfaceIndex x [[type("buffer_t")]],
                                       SurfaceIndex y [[type("buffer_t")]], uint n) {
  constexpr int S = 16; // LSC SIMD width on DG2
  uint first = thread_id() * LB;
  vector<float, LB> v;
  vector<uint, S> lane;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    lane(s) = s;
  }
#pragma unroll
  for (int i = 0; i < LB / S; ++i) {
    vector<uint, S> src = (n - 1 - first - i * S - lane) * 4; // byte offsets
    v.select<S, 1>(i * S) = cm_load<float>(x, src);
  }
  cm_store<float, LB>(y, first * 4, v);
}

// 256-bin histogram of bytes: each thread takes 64 DWords (256 bytes) and
// issues SIMD16 increments.
extern "C" _GENX_MAIN_ void hist_lsc(SurfaceIndex data [[type("buffer_t")]],
                                     SurfaceIndex bins [[type("buffer_t")]]) {
  constexpr int S = 16;
  vector<uint, LB> w = cm_load<uint, LB>(data, thread_id() * LB * 4);
  vector<uchar, LB * 4> b = w.format<uchar>();
#pragma unroll
  for (int i = 0; i < LB * 4 / S; ++i) {
    vector<uint, S> off = vector<uint, S>(b.select<S, 1>(i * S)) * 4; // bytes
    cm_atomic<AtomicOp::IINC, uint>(bins, off);
  }
}
