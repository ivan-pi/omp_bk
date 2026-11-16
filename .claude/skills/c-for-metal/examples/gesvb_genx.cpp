// gesvb_genx.cpp -- batched small dense solves with partial pivoting, the
// "batch in lanes" mapping (cuSolverDx's Thread operator: one SIMT lane per
// system). One CM thread solves L systems at once; lane s = system s.
//
// Same memory layout and results as gesv_genx.cpp (per system, column-major,
// back to back: A -> L/U, IPIV 1-based, B -> X), so the same host checks it.
//
// Registers are structure-of-arrays: a.row(c*N + r) = A(r,c) of all L systems.
// Every operation is a full SIMD-L instruction whatever N is -- the point for
// small N, where one system per thread would leave lanes idle.
//   * loads/stores: one L-lane gather/scatter per matrix element (lane s
//     addresses system s), no transposes;
//   * pivot search: a running compare + merge down the column gives each
//     lane its own max and index (first maximum, like LAPACK isamax);
//   * row swap: each lane has a DIFFERENT pivot row, so the swap is a
//     per-lane merge over the candidate rows -- the price of divergent data
//     in SIMD. Candidates no lane chose are skipped with a uniform any();
//   * elimination and solves: plain SIMD-L mads on whole rows of `a`.
// Tail batch: lanes beyond the end duplicate the last system (clamped
// addresses) and write identical values, so no mask is needed.
// Register budget: N*N*L floats (twice that while transposing) -- N <= 5
// with L = 16 fits 128 GRFs, N = 6..8 need -double-GRF.
//
// As a stand-alone kernel this (like gesv_genx.cpp) is memory-bound: each
// system crosses DRAM twice for 50-700 flops. Its real value is as an
// inline building block of a longer "mega-kernel" (factor_step/back_step
// are _GENX_ inline): build the small systems in registers, solve them in
// registers, consume the result in registers.
#include <cm/cm.h>

#ifndef GESV_N
#define GESV_N 4
#endif
#ifndef GESV_NRHS
#define GESV_NRHS 1
#endif
#ifndef GESVB_L
#define GESVB_L 16 // systems per thread (SIMD width of every operation)
#endif
constexpr int N = GESV_N;
constexpr int NR = GESV_NRHS;
constexpr int L = GESVB_L;
constexpr int NN = N * N;
static_assert(N >= 1 && N <= 8 && L >= 8 && L <= 32, "small N, 8..32 lanes");

// Loads/stores: the L systems of one thread are CONTIGUOUS in memory, so the
// fast path reads them with block messages and transposes in registers
// (array of structures -> structure of arrays), and the reverse on store.
// Gathers (one L-lane message per element, lane s -> system s) touch L
// cache lines per message and capped this kernel at ~225 GB/s for N >= 4.
// The blocked path holds a second copy of the data in registers: N <= 5 fits
// 128 GRFs, N = 6..8 need -double-GRF. -DGESVB_BLOCKED=0 selects gathers.
// Measured on an Arc A770, 1M systems, 1 rhs (ms): N=2 0.099 (gathers 0.149, one system per
// thread 0.373), N=3 0.224, N=4 0.381, N=5 0.606, N=6 0.825, N=7 1.176,
// N=8 1.431 (one system per thread 1.375: the crossover is at N ~ 8).
#ifndef GESVB_BLOCKED
#define GESVB_BLOCKED 1
#endif

// Element (r, c) of A as a row of `a`; element (r, j) of B as a row of `b`.
#define AE(r, c) a.row((c) * N + (r))
#define BE(r, j) b.row((j) * N + (r))

// Swap rows k and p(lane) of a SoA matrix with `cols` columns, per lane.
template <int K, int COLS, int ROWS>
_GENX_ inline void swap_rows(matrix_ref<float, ROWS, L> m, vector_ref<uint, L> p) {
#pragma unroll
  for (int i = K + 1; i < N; ++i) {
    vector<ushort, L> take = p == uint(i);
    if (take.any()) { // uniform: skip candidates no lane picked
#pragma unroll
      for (int c = 0; c < COLS; ++c) {
        vector<float, L> rk = m.row(c * N + K);
        m.row(c * N + K).merge(m.row(c * N + i), take);
        m.row(c * N + i).merge(rk, take);
      }
    }
  }
}

template <int K>
_GENX_ inline void factor_step(matrix_ref<float, NN, L> a, matrix_ref<float, N * NR, L> b,
                               matrix_ref<int, N, L> ipiv) {
  if constexpr (K < N) {
    // Per-lane pivot search over rows K..N-1 of column K.
    vector<float, L> best = cm_abs<float>(vector<float, L>(AE(K, K)));
    vector<uint, L> p = K;
#pragma unroll
    for (int i = K + 1; i < N; ++i) {
      vector<float, L> mag = cm_abs<float>(vector<float, L>(AE(i, K)));
      vector<ushort, L> gt = mag > best; // strict: first maximum wins
      best.merge(mag, gt);
      p.merge(uint(i), gt);
    }
    ipiv.row(K) = vector<int, L>(p) + 1;

    if (!(p == uint(K)).all()) {
      swap_rows<K, N, NN>(a, p);
      swap_rows<K, NR, N * NR>(b, p);
    }

    vector<float, L> rinv = 1.0f / vector<float, L>(AE(K, K));
#pragma unroll
    for (int i = K + 1; i < N; ++i) {
      AE(i, K) *= rinv; // multipliers
    }
#pragma unroll
    for (int j = K + 1; j < N; ++j) { // trailing update, one SIMD-L mad per element
#pragma unroll
      for (int i = K + 1; i < N; ++i) {
        AE(i, j) -= AE(i, K) * AE(K, j);
      }
    }
#pragma unroll
    for (int j = 0; j < NR; ++j) { // forward elimination of B
#pragma unroll
      for (int i = K + 1; i < N; ++i) {
        BE(i, j) -= AE(i, K) * BE(K, j);
      }
    }
    factor_step<K + 1>(a, b, ipiv);
  }
}

template <int K>
_GENX_ inline void back_step(matrix_ref<float, NN, L> a, matrix_ref<float, N * NR, L> b) {
  if constexpr (K >= 0) {
    vector<float, L> rinv = 1.0f / vector<float, L>(AE(K, K));
#pragma unroll
    for (int j = 0; j < NR; ++j) {
      BE(K, j) *= rinv;
#pragma unroll
      for (int i = 0; i < K; ++i) {
        BE(i, j) -= AE(i, K) * BE(K, j);
      }
    }
    back_step<K - 1>(a, b);
  }
}

// Blocked: ROWS elements of each of the L contiguous systems starting at
// element `first`. Chunks of 32 floats (16 when L*ROWS is not a multiple
// of 32); out-of-bounds reads give 0 and out-of-bounds writes are dropped,
// so the last, partial thread needs no clamping on this path.
template <int TOTAL>
constexpr int chunk_of() { return TOTAL % 32 == 0 ? 32 : TOTAL % 16 == 0 ? 16 : TOTAL % 8 == 0 ? 8 : 4; }

template <typename T, int ROWS>
_GENX_ inline void load_blocked(SurfaceIndex buf, uint first, matrix_ref<T, ROWS, L> m) {
  constexpr int TOTAL = L * ROWS;
  constexpr int C = chunk_of<TOTAL>();
  vector<T, TOTAL> raw;
#pragma unroll
  for (int r = 0; r < TOTAL / C; ++r) {
    read(buf, (first + C * r) * 4, raw.template select<C, 1>(C * r));
  }
  matrix_ref<T, L, ROWS> aos = raw.template format<T, L, ROWS>(); // row s = system s
#pragma unroll
  for (int e = 0; e < ROWS; ++e) {
    m.row(e) = aos.column(e); // transpose: element e of all systems
  }
}

template <typename T, int ROWS>
_GENX_ inline void store_blocked(SurfaceIndex buf, uint first, matrix_ref<T, ROWS, L> m) {
  constexpr int TOTAL = L * ROWS;
  constexpr int C = chunk_of<TOTAL>();
  vector<T, TOTAL> raw;
  matrix_ref<T, L, ROWS> aos = raw.template format<T, L, ROWS>();
#pragma unroll
  for (int e = 0; e < ROWS; ++e) {
    aos.column(e) = m.row(e);
  }
#pragma unroll
  for (int r = 0; r < TOTAL / C; ++r) {
    write(buf, (first + C * r) * 4, raw.template select<C, 1>(C * r));
  }
}

// Gather / scatter one element of every lane's system (element units).
template <typename T, int ROWS>
_GENX_ inline void gather_rows(SurfaceIndex buf, vector_ref<uint, L> base,
                               matrix_ref<T, ROWS, L> m) {
#pragma unroll
  for (int e = 0; e < ROWS; ++e) {
    vector<uint, L> off = base + e;
    vector_ref<T, L> row = m.row(e);
    read(buf, 0, off, row);
  }
}

template <typename T, int ROWS>
_GENX_ inline void scatter_rows(SurfaceIndex buf, vector_ref<uint, L> base,
                                matrix_ref<T, ROWS, L> m) {
#pragma unroll
  for (int e = 0; e < ROWS; ++e) {
    vector<uint, L> off = base + e;
    vector<T, L> row = m.row(e);
    write(buf, 0, off, row);
  }
}

extern "C" _GENX_MAIN_ void gesv_lanes(SurfaceIndex A [[type("buffer_t")]],
                                       SurfaceIndex B [[type("buffer_t")]],
                                       SurfaceIndex IPIV [[type("buffer_t")]], uint batch) {
  uint t = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);

  // Lane s handles system min(t*L + s, batch - 1).
  vector<uint, L> sys;
#pragma unroll
  for (int s = 0; s < L; ++s) {
    sys(s) = t * L + s;
  }
  sys.merge(batch - 1, sys >= batch);

  matrix<float, NN, L> a;
  matrix<float, N * NR, L> b;
  matrix<int, N, L> ipiv;
  vector<uint, L> abase = sys * NN;
  vector<uint, L> bbase = sys * (N * NR);
  vector<uint, L> pbase = sys * N;
  if constexpr (GESVB_BLOCKED) {
    load_blocked<float, NN>(A, t * L * NN, a);
    load_blocked<float, N * NR>(B, t * L * N * NR, b);
  } else {
    gather_rows<float, NN>(A, abase, a);
    gather_rows<float, N * NR>(B, bbase, b);
  }

  factor_step<0>(a, b, ipiv);
  back_step<N - 1>(a, b);

  if constexpr (GESVB_BLOCKED) {
    store_blocked<float, NN>(A, t * L * NN, a);
    store_blocked<float, N * NR>(B, t * L * N * NR, b);
    store_blocked<int, N>(IPIV, t * L * N, ipiv);
  } else {
    scatter_rows<float, NN>(A, abase, a);
    scatter_rows<float, N * NR>(B, bbase, b);
    scatter_rows<int, N>(IPIV, pbase, ipiv);
  }
}
