// gesv_genx.cpp -- batched small dense solves A X = B with partial pivoting
// (LAPACK sgesv semantics: getrf + getrs), one CM thread per system, the
// whole system in registers. Modelled on NVIDIA's cuSolverDx sample
// gesv_batched_partial_pivot_thread.cu.
//
// Memory (per system, column-major like LAPACK, systems back to back, any N):
//   A    N x N floats, lda = N    -> on return: L (unit, below the diagonal)
//                                    and U (on and above), as sgetrf
//   B    N x NRHS floats, ldb = N -> on return: the solution X
//   IPIV N ints                   -> 1-based pivot rows, as sgetrf
//
// Registers. A column-major matrix loaded as is gives At = A^T row-major:
// register row j of `at` is column j of A. Hence
//   * the pivot search over column k of A is a contiguous register row;
//   * swapping rows k and p of A swaps register COLUMNS k and p of `at`
//     (p is only known at run time -> indirect register addressing, the
//     GPU's "shuffle");
//   * the trailing update A(i,j) -= A(i,k) A(k,j), i,j > k, works on the
//     exactly sized (N-k-1)^2 block at.select<R,1,R,1>(k+1,k+1), either as
//     ONE outer-product mad built with replicate (default), or row by row
//     (GESV_ROWWISE=1: R mads of a scalar times a row -- no big temporaries,
//     for large N).
// The k loop is unrolled at compile time (if constexpr recursion), so every
// region has an exact constant size: the trailing work shrinks chunk by
// chunk with no wasted lanes.
//
// Any N: when N*N, N*NRHS and N are multiples of the block sizes, loads and
// stores are 128-byte block messages; otherwise (odd N, e.g. 7 or 13) the
// systems are only 4-byte aligned, so loads are DWord-aligned block reads
// (over-reading into the next system is harmless) and stores are exact
// scattered writes split by binary decomposition (49 = 32 + 16 + 1).
//
// Large N: N = 32 holds A in 128 GRFs; compile with the large register file
// (-double-GRF: 256 GRFs, half the threads per EU) and GESV_ROWWISE=1.
// B is eliminated alongside A (same swaps, same multipliers, same order as
// getrs' forward solve), then back-substituted. Divisions use the reciprocal
// of the pivot. Non-singular A is assumed (no zero-pivot check).
#include <cm/cm.h>

#ifndef GESV_N
#define GESV_N 16
#endif
#ifndef GESV_NRHS
#define GESV_NRHS 4
#endif
#ifndef GESV_ROWWISE
#define GESV_ROWWISE 0
#endif
constexpr int N = GESV_N;
constexpr int NR = GESV_NRHS;
static_assert(N >= 1 && N <= 32 && NR >= 1, "one system per thread: N <= 32");

// Register row pitch. When N is not a multiple of 8 floats (one 32-byte GRF),
// rows are padded to the next power of two (8, 16 or 32) so that every
// register row starts on a GRF boundary: unaligned rows (e.g. 31 floats =
// 124 bytes) make every region and the indirect swap straddle registers
// and spilled heavily for N = 31. Padded rows are loaded column by column.
constexpr int pow2ceil(int n) { return n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : 32; }
constexpr int NP = (N % 8 == 0) ? N : pow2ceil(N);
constexpr bool PADDED = NP != N;

// ---- loads and stores for any size ---------------------------------------------

constexpr int round32(int s) { return (s + 31) / 32 * 32; }

// Load S elements starting at element `base`.
template <typename T, int S>
_GENX_ inline void load(SurfaceIndex buf, uint base, vector_ref<T, S> v) {
  if constexpr (S % 32 == 0) { // every system starts 128-byte aligned: block reads
#pragma unroll
    for (int r = 0; r < S / 32; ++r) {
      read(buf, (base + 32 * r) * 4, v.template select<32, 1>(32 * r));
    }
  } else { // only 4-byte aligned: DWord-aligned block reads, over-reading
    vector<T, round32(S)> pad;
#pragma unroll
    for (int r = 0; r < round32(S) / 32; ++r) {
      read(DWALIGNED(buf), (base + 32 * r) * 4, pad.template select<32, 1>(32 * r));
    }
    v = pad.template select<S, 1>(0);
  }
}

// Exact scattered store of S elements of v from element OFF on (binary split).
template <typename T, int S, int OFF, int TOTAL>
_GENX_ inline void store_scattered(SurfaceIndex buf, uint base, vector_ref<T, TOTAL> v) {
  if constexpr (S > 0) {
    constexpr int K = S >= 32 ? 32 : S >= 16 ? 16 : S >= 8 ? 8 : S >= 4 ? 4 : S >= 2 ? 2 : 1;
    if constexpr (K == 1) {
      write(buf, base + OFF, 0u, v(OFF)); // scalar scattered write
    } else {
      vector<uint, K> off;
#pragma unroll
      for (int i = 0; i < K; ++i) {
        off(i) = i;
      }
      vector<T, K> d = v.template select<K, 1>(OFF);
      write(buf, base + OFF, off, d); // offsets in elements
    }
    store_scattered<T, S - K, OFF + K, TOTAL>(buf, base, v);
  }
}

// Store S elements at element `base`: block writes when aligned, else exact scatter.
template <typename T, int S>
_GENX_ inline void store(SurfaceIndex buf, uint base, vector_ref<T, S> v) {
  if constexpr (S % 32 == 0) {
#pragma unroll
    for (int r = 0; r < S / 32; ++r) {
      write(buf, (base + 32 * r) * 4, v.template select<32, 1>(32 * r));
    }
  } else if constexpr (S == 4 || S == 8 || S == 16) { // 1, 2 or 4 OWords; base is a multiple of S
    write(buf, base * 4, v);
  } else {
    store_scattered<T, S, 0, S>(buf, base, v);
  }
}

// Padded layout: ROWS register rows of NP floats <-> ROWS memory columns of N.
template <int ROWS>
_GENX_ inline void load_padded(SurfaceIndex buf, uint base, matrix_ref<float, ROWS, NP> m) {
#pragma unroll
  for (int j = 0; j < ROWS; ++j) { // one DWord-aligned block read per column (over-reads)
    vector_ref<float, NP> row = m.row(j);
    read(DWALIGNED(buf), (base + j * N) * 4, row);
  }
}

template <int ROWS>
_GENX_ inline void store_padded(SurfaceIndex buf, uint base, matrix_ref<float, ROWS, NP> m) {
  // W depends on ROWS so that this body is only checked when instantiated
  // (for N = 24, NP = 24 is not a legal scatter width, but that layout is
  // never padded).
  constexpr int W = ROWS > 0 ? NP : NP;
  if constexpr (PADDED && ROWS > 0) {
    vector<uint, W> off; // padding lanes rewrite element N-1 with its own value
#pragma unroll
    for (int i = 0; i < W; ++i) {
      off(i) = i < N ? i : N - 1;
    }
#pragma unroll
    for (int j = 0; j < ROWS; ++j) { // one scattered write per column
      vector<float, W> d = m.row(j);
      d.merge(d(N - 1), vector<ushort, W>(off) == ushort(N - 1));
      write(buf, base + j * N, off, d);
    }
  }
}

// Load/store ROWS register rows (= memory columns). Templates, so that only
// the layout in use is instantiated: `if constexpr` in a non-template
// function still type-checks the discarded branch.
template <int ROWS>
_GENX_ inline void load_rows(SurfaceIndex buf, uint base, matrix_ref<float, ROWS, NP> m) {
  if constexpr (PADDED) {
    load_padded<ROWS>(buf, base, m);
  } else {
    load<float, ROWS * NP>(buf, base, m.template format<float>());
  }
}

template <int ROWS>
_GENX_ inline void store_rows(SurfaceIndex buf, uint base, matrix_ref<float, ROWS, NP> m) {
  if constexpr (PADDED) {
    store_padded<ROWS>(buf, base, m);
  } else {
    store<float, ROWS * NP>(buf, base, m.template format<float>());
  }
}

// ---- the factorization -----------------------------------------------------------

// Column k of A is register row k of at; find the pivot row p >= k.
template <int K>
_GENX_ inline uint pivot_row(matrix_ref<float, N, NP> at) {
  vector<float, NP> mag = cm_abs<float>(vector<float, NP>(at.row(K)));
  vector<uint, NP> lane;
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    lane(i) = i;
  }
  mag.merge(-1.0f, (lane < K) | (lane >= N)); // above the diagonal or padding: no
  float best = cm_reduced_max<float>(mag);
  vector<uint, NP> cand = N;
  cand.merge(lane, mag == best);
  return cm_reduced_min<uint>(cand); // first maximum, like LAPACK isamax
}

// Swap register columns k and p (rows k and p of A or B).
template <int K, int ROWS>
_GENX_ inline void swap_columns(matrix_ref<float, ROWS, NP> m, uint p) {
  vector<float, ROWS> ck = m.column(K);
  vector<float, ROWS> cp = m.column(p); // runtime column: indirect access
  m.column(p) = ck;
  m.column(K) = cp;
}

// One elimination step k, then k + 1, ... (compile-time recursion).
template <int K>
_GENX_ inline void factor_step(matrix_ref<float, N, NP> at, matrix_ref<float, NR, NP> bt,
                               vector_ref<int, N> ipiv) {
  if constexpr (K < N) {
    uint p = pivot_row<K>(at);
    ipiv(K) = int(p) + 1;
    if (p != K) { // uniform branch: same for the whole thread
      swap_columns<K, N>(at, p);
      swap_columns<K, NR>(bt, p);
    }

    constexpr int R = N - K - 1;
    if constexpr (R > 0) {
      float rinv = 1.0f / at(K, K);
      // Multipliers: A(i,k) /= A(k,k) for i > k  (register row k, columns > k).
      vector<float, R> l = at.row(K).template select<R, 1>(K + 1) * rinv;
      at.row(K).template select<R, 1>(K + 1) = l;

      // Trailing update: at(j,i) -= A(k,j) * l(i) for j, i > k, with
      // A(k,j) = register column k (rows j > k).
#if GESV_ROWWISE
#pragma unroll
      for (int j = K + 1; j < N; ++j) {
        at.row(j).template select<R, 1>(K + 1) -= at(j, K) * l; // scalar x row
      }
#else
      vector<float, R> u = at.template select<R, 1, 1, 1>(K + 1, K);
      at.template select<R, 1, R, 1>(K + 1, K + 1) -=
          u.template replicate<R, 1, R, 0>(0) * l.template replicate<R>();
#endif

      // Same elimination on the right-hand sides: B(i,r) -= l(i) * B(k,r).
#pragma unroll
      for (int r = 0; r < NR; ++r) {
        bt.row(r).template select<R, 1>(K + 1) -= bt(r, K) * l;
      }
    }
    factor_step<K + 1>(at, bt, ipiv);
  }
}

// Back substitution with U, k = N-1 down to 0.
template <int K>
_GENX_ inline void back_step(matrix_ref<float, N, NP> at, matrix_ref<float, NR, NP> bt) {
  if constexpr (K >= 0) {
    float rinv = 1.0f / at(K, K);
    vector<float, NR> xk = bt.column(K) * rinv;
    bt.column(K) = xk;
    if constexpr (K > 0) {
      // B(i,r) -= U(i,k) x_k(r) for i < k; U(i,k) = register row k, column i.
      vector<float, K> ucol = at.row(K).template select<K, 1>(0);
#pragma unroll
      for (int r = 0; r < NR; ++r) {
        bt.row(r).template select<K, 1>(0) -= xk(r) * ucol;
      }
    }
    back_step<K - 1>(at, bt);
  }
}

extern "C" _GENX_MAIN_ void gesv_batched(SurfaceIndex A [[type("buffer_t")]],
                                         SurfaceIndex B [[type("buffer_t")]],
                                         SurfaceIndex IPIV [[type("buffer_t")]], uint batch) {
  uint s = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  if (s >= batch) {
    return;
  }

  // Load: column-major A -> at (= A^T row-major), column-major B -> bt.
  matrix<float, N, NP> at;
  matrix<float, NR, NP> bt;
  load_rows<N>(A, s * N * N, at);
  load_rows<NR>(B, s * N * NR, bt);

  // Factor (with the right-hand sides eliminated alongside) and solve.
  vector<int, N> ipiv;
  factor_step<0>(at, bt, ipiv);
  back_step<N - 1>(at, bt);

  // Store the LU factors, pivots and solution.
  store_rows<N>(A, s * N * N, at);
  store_rows<NR>(B, s * N * NR, bt);
  store<int, N>(IPIV, s * N, ipiv);
}
