// patterns_genx.cpp -- C for Metal idioms, each kernel checked by
// host_patterns.cpp against a CPU reference.
//
//   transpose8   2D thread space, matrix select/format/column, in-register transpose
//   hist256      byte data -> scattered atomic increments (write_atomic<ATOMIC_INC>)
//   clamp_scale  SIMD control flow (SIMD_IF_BEGIN / SIMD_ELSEIF / SIMD_ELSE) vs. merge
//   chunk_sum    cm_sum reduction, scalar scattered write of one result per thread
//   reverse32    iselect (register-indirect gather) and replicate
//   group_sum    SLM + barrier: threads of one group combine partial sums
//   vadd_svm     stateless (SVM / USM pointer) access with cm_svm_block_read/write
//
// All kernels use the legacy dataport API, which every Gen9+ target accepts;
// the LSC API (cm_load/cm_ptr_load, DG2+) is in lsc_genx.cpp.
#include <cm/cm.h>

// ---------------------------------------------------------------------------
// transpose8: B = A^T for an n x n float matrix, one 8x8 tile per thread.
// Thread space is 2D: (tile column, tile row) = (cm_group_id(0), cm_group_id(1)).
extern "C" _GENX_MAIN_ void transpose8(SurfaceIndex A [[type("buffer_t")]],
                                       SurfaceIndex B [[type("buffer_t")]],
                                       uint n) {
  uint tj = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint ti = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);

  matrix<float, 8, 8> tile;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    // A row segment of 8 floats is 32 bytes = 2 OWords; offset is 16B aligned
    // because n is a multiple of 8.
    vector_ref<float, 8> row = tile.row(r);
    read(A, ((ti * 8 + r) * n + tj * 8) * sizeof(float), row);
  }

  // Transpose in registers. column(c) is a strided region (stride 8), so
  // each assignment is a single regioned mov -- no memory traffic.
  matrix<float, 8, 8> t;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    t.row(c) = tile.column(c);
  }

  // Equivalent formulation through a 1D view: element (r, c) of the
  // transposed tile is element c*8 + r of the flattened source.
  // vector_ref<float, 64> flat = tile.format<float>();
  // t.row(c) = flat.select<8, 8>(c);

#pragma unroll
  for (int r = 0; r < 8; ++r) {
    write(B, ((tj * 8 + r) * n + ti * 8) * sizeof(float), t.row(r));
  }
}

// ---------------------------------------------------------------------------
// hist256: count byte values. Each thread reads 32 bytes and issues
// atomic-increment messages; lanes hitting the same bin are still counted
// correctly (each lane's update is atomic).
// Use SIMD16 (or SIMD8) atomics: a single SIMD32 write_atomic lost half of
// the increments on Gen12LP (UHD 750) and hangs DG2 unless the backend gets
// -ftranslate-legacy-memory-intrinsics. SIMD16 works on both without it.
extern "C" _GENX_MAIN_ void hist256(SurfaceIndex data [[type("buffer_t")]],
                                    SurfaceIndex hist [[type("buffer_t")]]) {
  uint tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  vector<uchar, 32> px;
  read(data, tid * 32, px);
  vector<uint, 32> bin = px; // implicit uchar -> uint conversion
  vector<uint, 16> lo = bin.select<16, 1>(0);
  vector<uint, 16> hi = bin.select<16, 1>(16);
  write_atomic<ATOMIC_INC, uint>(hist, lo); // offsets are in DWords
  write_atomic<ATOMIC_INC, uint>(hist, hi);
}

// ---------------------------------------------------------------------------
// clamp_scale: y = x < 0 ? 0 : (x > 10 ? x / 2 : x + 1)
// Shown twice: with SIMD control flow (per-lane branching, hardware masks)
// and with compare + merge (predicated moves, no branches).
extern "C" _GENX_MAIN_ void clamp_scale(SurfaceIndex in [[type("buffer_t")]],
                                        SurfaceIndex out_cf [[type("buffer_t")]],
                                        SurfaceIndex out_merge [[type("buffer_t")]]) {
  uint tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint off = tid * 32 * sizeof(float);
  vector<float, 32> x;
  read(in, off, x);

  vector<float, 32> y = x;
  SIMD_IF_BEGIN(x < 0.0f) {
    y = 0.0f;
  }
  SIMD_ELSEIF(x > 10.0f) {
    y = x * 0.5f;
  }
  SIMD_ELSE {
    y = x + 1.0f;
  }
  SIMD_IF_END;
  write(out_cf, off, y);

  vector<float, 32> z = x + 1.0f;
  z.merge(x * 0.5f, x > 10.0f); // lanes with x > 10 take x/2
  z.merge(0.0f, x < 0.0f);      // lanes with x < 0 take 0
  write(out_merge, off, z);
}

// ---------------------------------------------------------------------------
// chunk_sum: out[tid] = sum of a 32-float chunk.
extern "C" _GENX_MAIN_ void chunk_sum(SurfaceIndex in [[type("buffer_t")]],
                                      SurfaceIndex out [[type("buffer_t")]]) {
  uint tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  vector<float, 32> x;
  read(in, tid * 32 * sizeof(float), x);
  float s = cm_sum<float>(x);
  // Scalar scattered write: (global_offset, element_offset) in element units.
  write(out, tid, 0u, s);
}

// ---------------------------------------------------------------------------
// reverse32: out = reverse(in) per 32-int chunk with iselect, plus a
// replicate demo: rep = {x0,x0,x0,x0, x4,x4,x4,x4, ...} for the first 8 groups.
extern "C" _GENX_MAIN_ void reverse32(SurfaceIndex in [[type("buffer_t")]],
                                      SurfaceIndex out [[type("buffer_t")]],
                                      SurfaceIndex rep [[type("buffer_t")]]) {
  uint tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint off = tid * 32 * sizeof(int);
  vector<int, 32> x;
  read(in, off, x);

  vector<ushort, 32> idx;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    idx(i) = 31 - i;
  }
  vector<int, 32> r = x.iselect(idx); // r(i) = x(idx(i)); r-value only
  write(out, off, r);

  // replicate<REP, VS, W, HS>(i): REP blocks of W elements, block starts VS
  // apart, elements HS apart. <8, 4, 4, 0> -> 8 blocks, each 4 copies of x(4k).
  vector<int, 32> rp = x.replicate<8, 4, 4, 0>(0);
  write(rep, off, rp);
}

// ---------------------------------------------------------------------------
// group_sum: GS threads per group each sum a 32-float chunk, publish the
// partial sum through SLM, and thread 0 of the group writes the group total.
constexpr int GS = 8; // threads per group; host must use local size GS

extern "C" _GENX_MAIN_ void group_sum(SurfaceIndex in [[type("buffer_t")]],
                                      SurfaceIndex out [[type("buffer_t")]]) {
  cm_slm_init(GS * 16);              // bytes of SLM per group
  uint slm = cm_slm_alloc(GS * 16);  // a handle; offsets go in the offset argument

  uint lid = cm_local_id(0);
  uint gid = cm_group_id(0);
  uint tid = gid * GS + lid;

  vector<float, 32> x;
  read(in, tid * 32 * sizeof(float), x);

  vector<float, 4> part = 0.0f; // SLM block writes are OWord (16 B) granular
  part(0) = cm_sum<float>(x);
  cm_slm_block_write(slm, lid * 16, part);

  cm_slm_fence(CM_GLOBAL_COHERENT_FENCE); // make the write visible ...
  cm_barrier();                           // ... before any thread passes here

  if (lid == 0) {
    vector<float, GS * 4> all;
    cm_slm_block_read(slm, GENX_NONE, 0, all);
    float total = cm_sum<float>(all.select<GS, 4>(0)); // every 4th element
    write(out, gid, 0u, total);
  }
}

// ---------------------------------------------------------------------------
// vadd_svm: stateless variant of vadd. svmptr_t arguments are raw 64-bit
// addresses (OpenCL SVM / USM pointers); no surface state, no 4 GB limit.
extern "C" _GENX_MAIN_ void vadd_svm(svmptr_t a [[type("svmptr_t")]],
                                     svmptr_t b [[type("svmptr_t")]],
                                     svmptr_t c [[type("svmptr_t")]]) {
  uint tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  svmptr_t off = (svmptr_t)tid * 32 * sizeof(float); // svmptr_t is a byte address
  vector<float, 32> x, y;
  cm_svm_block_read(a + off, x); // address must be 16-byte aligned
  cm_svm_block_read(b + off, y);
  cm_svm_block_write(c + off, vector<float, 32>(x + y));
}
