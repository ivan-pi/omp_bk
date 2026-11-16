// argmax_genx.cpp -- index of the maximum of a large int32 or double array
// (first occurrence on ties): a bandwidth-bound two-stage reduction.
//
// Stage 1 (argmax_*_p1): every CM thread scans ARGMAX_CHUNK contiguous
//   elements. Per iteration it issues ARGMAX_U block reads of 128 B back to
//   back, then updates per-lane running (max, index) registers with compare +
//   merge -- no branches; strict ">" keeps the earliest index in each lane.
//   Finally it reduces its lanes to one (max, smallest index) pair.
// Stage 2 (argmax_*_p2): a reduction TREE over the partials: each thread
//   reduces up to P_CHUNK partials with block reads; the host relaunches it
//   until one partial is left (131072 -> 128 -> 1 for 2^26 ints).
// Partials are stored as separate arrays (structure of arrays) in one buffer:
//   int32 : value[0..S), index[S..2S)        double: khi[..), klo[..), index[..)
// so stage 2 can use block reads instead of gathers.
//
// The last chunk masks lanes >= n: stateful out-of-bounds reads return 0,
// which would win against negative data.
//
// double on DG2 (no FP64 ALU): argmax needs only ORDER, not arithmetic. The
// 64-bit pattern maps to an unsigned key with the same order (negative:
// invert all bits; positive: flip the sign bit; -0.0 -> +0.0 so that equal
// values tie as in IEEE), compared as two 32-bit halves. NaNs order above
// +inf (sign clear) or below -inf (sign set), unlike IEEE.
//
// readsum_i32: plain read + sum with stage 1's access pattern -- the
// bandwidth reference (stage 1 should run at the same speed).
#include <cm/cm.h>

// Tuning knobs (override with CMC_FLAGS="-DARGMAX_CHUNK=... -DARGMAX_U=...").
// Short macro names such as -DU=4 break <cm/cm.h>, whose templates use U.
#ifndef ARGMAX_CHUNK
#define ARGMAX_CHUNK 512 // elements per thread in stage 1
#endif
#ifndef ARGMAX_U
#define ARGMAX_U 2 // 128-byte reads in flight per iteration
#endif
constexpr int CHUNK = ARGMAX_CHUNK;
constexpr int U = ARGMAX_U;
constexpr int I_LANES = 32 * U; // int32 lanes per iteration (32 ints = 128 B per read)
constexpr int D_LANES = 16 * U; // double lanes per iteration (16 doubles = 128 B)
static_assert(CHUNK % I_LANES == 0 && CHUNK % D_LANES == 0, "chunk must be whole iterations");

constexpr int P_LANES = 128;   // stage 2: partials per iteration (4 block reads per array)
constexpr int P_CHUNK = 1024;  // stage 2: partials per thread
static_assert(P_CHUNK % P_LANES == 0, "stage-2 chunk must be whole iterations");

constexpr uint NONE = 0xFFFFFFFFu; // "no candidate yet" index

// ---- helpers ------------------------------------------------------------------

template <int N>
_GENX_ inline vector<uint, N> iota(uint start) {
  vector<uint, N> v;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    v(i) = start + i;
  }
  return v;
}

// Lanes -> (max value, smallest index holding it).
template <int N>
_GENX_ inline uint first_index_of_max(vector_ref<int, N> best, vector_ref<uint, N> bidx,
                                      int &value) {
  value = cm_reduced_max<int>(best);
  vector<uint, N> cand = NONE;
  cand.merge(bidx, (best == value) & (bidx != NONE));
  return cm_reduced_min<uint>(cand);
}

template <int N>
_GENX_ inline uint first_index_of_max_key(vector_ref<uint, N> bhi, vector_ref<uint, N> blo,
                                          vector_ref<uint, N> bidx, uint &khi, uint &klo) {
  vector<uint, N> hi = 0;
  hi.merge(bhi, bidx != NONE);
  khi = cm_reduced_max<uint>(hi);
  vector<ushort, N> top = (bhi == khi) & (bidx != NONE);
  vector<uint, N> lo = 0;
  lo.merge(blo, top);
  klo = cm_reduced_max<uint>(lo);
  vector<uint, N> cand = NONE;
  cand.merge(bidx, top & (blo == klo));
  return cm_reduced_min<uint>(cand);
}

// double bit patterns (lo, hi word pairs) -> order-preserving unsigned keys.
template <int N>
_GENX_ inline void double_keys(vector_ref<uint, 2 * N> raw, vector_ref<uint, N> khi,
                               vector_ref<uint, N> klo) {
  vector<uint, N> hi = raw.template select<N, 2>(1); // little endian: high word second
  vector<uint, N> lo = raw.template select<N, 2>(0);
  hi.merge(0u, (hi == 0x80000000u) & (lo == 0u));    // -0.0 -> +0.0
  vector<ushort, N> neg = (hi & 0x80000000u) != 0u;
  khi = hi ^ 0x80000000u;                            // positive: flip the sign bit
  klo = lo;
  khi.merge(~hi, neg);                               // negative: invert everything
  klo.merge(~lo, neg);
}

// Block read of N uints (N a multiple of 32) starting at element `at`.
template <int N>
_GENX_ inline void read_block(SurfaceIndex buf, uint at, vector_ref<uint, N> v) {
#pragma unroll
  for (int r = 0; r < N / 32; ++r) {
    read(buf, (at + 32 * r) * sizeof(uint), v.template select<32, 1>(32 * r));
  }
}

// ---- stage 1 ------------------------------------------------------------------

// PART layout: value[S], index[S]; S = stride (>= number of threads).
extern "C" _GENX_MAIN_ void argmax_i32_p1(SurfaceIndex A [[type("buffer_t")]],
                                          SurfaceIndex PART [[type("buffer_t")]], uint n,
                                          uint stride) {
  uint t = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint begin = t * CHUNK;

  vector<int, I_LANES> best = 0;
  vector<uint, I_LANES> bidx = NONE;
  vector<uint, I_LANES> idx = iota<I_LANES>(begin);
  bool full = begin + CHUNK <= n; // only the last thread sees the end

#pragma unroll
  for (int it = 0; it < CHUNK / I_LANES; ++it) {
    vector<int, I_LANES> v;
    read_block<I_LANES>(A, begin + it * I_LANES, v.format<uint>());

    // Take the element if it is larger, or if the lane has no candidate yet.
    vector<ushort, I_LANES> take = (v > best) | (bidx == NONE);
    if (!full) {
      take &= idx < n;
    }
    best.merge(v, take);
    bidx.merge(idx, take);
    idx += I_LANES;
  }

  int value;
  uint at = first_index_of_max<I_LANES>(best, bidx, value);
  write(PART, t, 0u, uint(value));
  write(PART, stride + t, 0u, at);
}

// PART layout: khi[S], klo[S], index[S].
extern "C" _GENX_MAIN_ void argmax_f64_p1(SurfaceIndex A [[type("buffer_t")]],
                                          SurfaceIndex PART [[type("buffer_t")]], uint n,
                                          uint stride) {
  uint t = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint begin = t * CHUNK;

  vector<uint, D_LANES> bhi = 0;
  vector<uint, D_LANES> blo = 0;
  vector<uint, D_LANES> bidx = NONE;
  vector<uint, D_LANES> idx = iota<D_LANES>(begin);
  bool full = begin + CHUNK <= n;

#pragma unroll
  for (int it = 0; it < CHUNK / D_LANES; ++it) {
    vector<uint, 2 * D_LANES> raw; // D_LANES doubles as (lo, hi) word pairs
    read_block<2 * D_LANES>(A, 2 * (begin + it * D_LANES), raw);
    vector<uint, D_LANES> khi, klo;
    double_keys<D_LANES>(raw, khi, klo);

    vector<ushort, D_LANES> take = (khi > bhi) | ((khi == bhi) & (klo > blo)) | (bidx == NONE);
    if (!full) {
      take &= idx < n;
    }
    bhi.merge(khi, take);
    blo.merge(klo, take);
    bidx.merge(idx, take);
    idx += D_LANES;
  }

  uint khi, klo;
  uint at = first_index_of_max_key<D_LANES>(bhi, blo, bidx, khi, klo);
  write(PART, t, 0u, khi);
  write(PART, stride + t, 0u, klo);
  write(PART, 2 * stride + t, 0u, at);
}

// ---- stage 2: reduction tree over partials ---------------------------------------

// IN holds `nparts` partials with stride sin; OUT gets one per thread, stride sout.
extern "C" _GENX_MAIN_ void argmax_i32_p2(SurfaceIndex IN [[type("buffer_t")]],
                                          SurfaceIndex OUT [[type("buffer_t")]], uint nparts,
                                          uint sin, uint sout) {
  uint g = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint begin = g * P_CHUNK;
  vector<int, P_LANES> best = 0;
  vector<uint, P_LANES> bidx = NONE;
  vector<uint, P_LANES> pos = iota<P_LANES>(begin);

  for (uint p = begin; p < begin + P_CHUNK && p < nparts; p += P_LANES) {
    vector<int, P_LANES> v;
    vector<uint, P_LANES> at;
    read_block<P_LANES>(IN, p, v.format<uint>());
    read_block<P_LANES>(IN, sin + p, at);
    // Partials are in index order, so strict ">" keeps the first occurrence.
    vector<ushort, P_LANES> take = ((v > best) | (bidx == NONE)) & (pos < nparts);
    best.merge(v, take);
    bidx.merge(at, take);
    pos += P_LANES;
  }

  int value;
  uint at = first_index_of_max<P_LANES>(best, bidx, value);
  write(OUT, g, 0u, uint(value));
  write(OUT, sout + g, 0u, at);
}

extern "C" _GENX_MAIN_ void argmax_f64_p2(SurfaceIndex IN [[type("buffer_t")]],
                                          SurfaceIndex OUT [[type("buffer_t")]], uint nparts,
                                          uint sin, uint sout) {
  uint g = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint begin = g * P_CHUNK;
  vector<uint, P_LANES> bhi = 0;
  vector<uint, P_LANES> blo = 0;
  vector<uint, P_LANES> bidx = NONE;
  vector<uint, P_LANES> pos = iota<P_LANES>(begin);

  for (uint p = begin; p < begin + P_CHUNK && p < nparts; p += P_LANES) {
    vector<uint, P_LANES> khi, klo, at;
    read_block<P_LANES>(IN, p, khi);
    read_block<P_LANES>(IN, sin + p, klo);
    read_block<P_LANES>(IN, 2 * sin + p, at);
    vector<ushort, P_LANES> take =
        ((khi > bhi) | ((khi == bhi) & (klo > blo)) | (bidx == NONE)) & (pos < nparts);
    bhi.merge(khi, take);
    blo.merge(klo, take);
    bidx.merge(at, take);
    pos += P_LANES;
  }

  uint khi, klo;
  uint at = first_index_of_max_key<P_LANES>(bhi, blo, bidx, khi, klo);
  write(OUT, g, 0u, khi);
  write(OUT, sout + g, 0u, klo);
  write(OUT, 2 * sout + g, 0u, at);
}

// ---- bandwidth reference ------------------------------------------------------------

extern "C" _GENX_MAIN_ void readsum_i32(SurfaceIndex A [[type("buffer_t")]],
                                        SurfaceIndex PART [[type("buffer_t")]], uint n,
                                        uint stride) {
  uint t = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint begin = t * CHUNK;
  vector<int, I_LANES> acc = 0;
#pragma unroll
  for (int it = 0; it < CHUNK / I_LANES; ++it) {
    vector<int, I_LANES> v;
    read_block<I_LANES>(A, begin + it * I_LANES, v.format<uint>());
    acc += v;
  }
  write(PART, t, 0u, uint(cm_sum<int>(acc)));
}
