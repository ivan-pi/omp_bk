// spmv_genx.cpp -- y = A x for a CSR matrix, the SpMV case study of the CM
// paper and CGO 2021 talk (Lueh et al.):
//
//   * one hardware thread per row;
//   * the row's values (ANZ) and column indices (ACOL) are loaded in slices
//     of N = 8, 16 or 32 with DWord-aligned block reads (a row starts at any
//     element, so the offset is only 4-byte aligned);
//   * lanes past the end of the row are pointed at column 0 with merge. The x
//     vector is stored with a ZERO PADDED IN FRONT (x_pad[0] = 0, x_pad[j+1] =
//     x[j]; ACOL holds column + 1), so those lanes gather 0 and contribute 0;
//   * x is gathered with a scattered read at the column indices;
//   * cm_dp4 computes 4-element dot products (each result replicated over its
//     4 lanes) and cm_sum of every 4th lane gives the slice's sum. This follows
//     the paper (written for Gen11, which had a dp4 instruction); Gen12LP and
//     DG2 have none, so the compiler emits mul/mad -- `cm_sum<float>(v_an * v_x)`
//     is equivalent there;
//   * the slice width is chosen PER ROW with ordinary scalar control flow --
//     the "dynamic SIMD width" the paper uses for very sparse rows: a row with
//     3 nonzeros costs one SIMD8 slice, not a SIMD32 one.
//
// Thread space: nrows threads (1D), local size 1.
#include <cm/cm.h>

// Sum of row entries [begin, end) using slices of N lanes.
template <int N>
_GENX_ inline float row_dot(SurfaceIndex ACOL, SurfaceIndex ANZ, SurfaceIndex X,
                            uint begin, uint end) {
  static_assert(N % 4 == 0 && N <= 32, "dp4 groups of 4; at most 128 B per block read");

  vector<uint, N> lane;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    lane(i) = i;
  }

  float sum = 0.0f;
  for (uint off = begin; off < end; off += N) {
    // Slice of column indices and values (may run past the row end).
    vector<uint, N> v_ac;
    vector<float, N> v_an;
    read(DWALIGNED(ACOL), off * sizeof(uint), v_ac);
    read(DWALIGNED(ANZ), off * sizeof(float), v_an);

    // Lanes past the row end -> column 0 (the padded zero of x) and value 0.
    vector<ushort, N> past = (lane + off) >= end;
    v_ac.merge(0u, past);
    v_an.merge(0.0f, past);

    // Gather x at the column indices (element offsets, units of float).
    vector<float, N> v_x;
    read(X, 0, v_ac, v_x);

    // 4-wide dot products; each group's result sits in all 4 of its lanes.
    vector<float, N> v_dp4 = cm_dp4<float>(v_an, v_x);
    sum += cm_sum<float>(v_dp4.template select<N / 4, 4>(0));
  }
  return sum;
}

extern "C" _GENX_MAIN_ void spmv_csr(SurfaceIndex ROWPTR [[type("buffer_t")]],
                                     SurfaceIndex ACOL [[type("buffer_t")]],
                                     SurfaceIndex ANZ [[type("buffer_t")]],
                                     SurfaceIndex X [[type("buffer_t")]],
                                     SurfaceIndex Y [[type("buffer_t")]]) {
  uint row = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);

  // rowptr[row] and rowptr[row + 1] with one 2-lane gather.
  vector<uint, 2> idx;
  idx(0) = 0;
  idx(1) = 1;
  vector<uint, 2> bounds;
  read(ROWPTR, row, idx, bounds);
  uint begin = bounds(0);
  uint end = bounds(1);
  uint len = end - begin;

  // Pick the slice width from the row length (uniform branch per thread).
  float y;
  if (len <= 8) {
    y = row_dot<8>(ACOL, ANZ, X, begin, end);
  } else if (len <= 16) {
    y = row_dot<16>(ACOL, ANZ, X, begin, end);
  } else {
    y = row_dot<32>(ACOL, ANZ, X, begin, end);
  }

  write(Y, row, 0u, y); // scalar scattered write, element units
}
