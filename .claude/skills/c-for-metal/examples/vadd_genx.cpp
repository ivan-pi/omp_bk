// vadd_genx.cpp -- minimal C for Metal kernel: c = a + b.
//
// One kernel invocation is ONE hardware thread. It owns a contiguous chunk of
// CHUNK floats and walks it in BLOCK-sized pieces, BLOCK being the largest
// single message (OWord block read/write: 128 bytes = 32 floats). The loop is
// unrolled, so every offset is a compile-time constant and each piece is one
// `send` straight into/out of a register region of the chunk.
// The host dispatches n / CHUNK threads with local size 1 (the "SIMD1"
// dispatch of the OpenCL / Level Zero runtimes).
//
// Build (see scripts/cm_build.sh):
//   cmc -march=dg2 -emit-spirv -o vadd_genx.spv vadd_genx.cpp
//   ocloc compile -file vadd_genx.spv -spirv_input -device dg2 \
//         -options "-vc-codegen" -output vadd_genx -output_no_suffix
//   (or straight to a binary: cmc -march=dg2 -o vadd_genx.bin vadd_genx.cpp)
#include <cm/cm.h>

#ifndef CHUNK
#define CHUNK 128 // floats per thread; host_ocl.cpp / host_sycl.cpp assume 128
#endif
constexpr int BLOCK = 32; // floats per OWord block message (128 B maximum)
static_assert(CHUNK % BLOCK == 0, "a chunk must be whole OWord blocks");
static_assert(CHUNK * 3 * sizeof(float) <= 64 * 32, "a, b and c chunks should use at most half the GRF");

extern "C" _GENX_MAIN_ void vadd(SurfaceIndex a [[type("buffer_t")]],
                                 SurfaceIndex b [[type("buffer_t")]],
                                 SurfaceIndex c [[type("buffer_t")]]) {
  // Thread id: with SIMD1 dispatch, global id == group id * local size + local id.
  uint tid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
  uint off = tid * CHUNK * sizeof(float); // byte offset, 16-byte aligned

  vector<float, CHUNK> x, y;
#pragma unroll
  for (int i = 0; i < CHUNK / BLOCK; ++i) {
    // OWord block read: offset 16-byte aligned, at most 128 bytes per call
    read(a, off + i * BLOCK * sizeof(float), x.select<BLOCK, 1>(i * BLOCK));
    read(b, off + i * BLOCK * sizeof(float), y.select<BLOCK, 1>(i * BLOCK));
  }
  x += y; // one SIMD-CHUNK statement; the compiler splits it into native widths
#pragma unroll
  for (int i = 0; i < CHUNK / BLOCK; ++i) {
    write(c, off + i * BLOCK * sizeof(float), x.select<BLOCK, 1>(i * BLOCK));
  }
}
