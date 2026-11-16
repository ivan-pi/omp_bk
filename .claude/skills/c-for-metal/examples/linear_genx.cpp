// linear_genx.cpp -- the 3x3 box ("linear") filter of the CM paper
// (Lueh et al., CGO 2021, Algorithm 2) and of the CM spec, Appendix A.
//
// The image is packed RGB, 3 bytes per pixel, stored as a 2D surface of
// 1-byte pixels (width = 3 * pixels). Each thread loads an 8x32-byte block
// with ONE 2D media-block read, builds the nine shifted 6x24 windows with
// select<> (pure register regioning, no extra loads) and writes a 6x24 block.
//
// Output(Y, X) = 0.111 * sum over dy in {0,1,2}, dx in {0,3,6} of In(Y+dy, X+dx)
// (reads past the right/bottom edge are clamped to the edge by the hardware).
//
// Thread space: (width_bytes / 24) x (height / 6), local size 1x1.
#include <cm/cm.h>

// Argument annotation for a 2D surface used with media block read/write.
// The 2021 cmc only knows "image2d_t" (which then means a media-block
// surface) and silently ignores "image2d_media_block_t" -- the argument then
// becomes a buffer and clSetKernelArg fails with CL_INVALID_MEM_OBJECT (-38).
// Newer cmc releases deprecate that meaning of "image2d_t" and expect
// "image2d_media_block_t"; override with -DCM_MEDIA_IMAGE='"image2d_media_block_t"'.
#ifndef CM_MEDIA_IMAGE
#define CM_MEDIA_IMAGE "image2d_t"
#endif

extern "C" _GENX_MAIN_ void linear(SurfaceIndex ibuf [[type(CM_MEDIA_IMAGE)]],
                                   SurfaceIndex obuf [[type(CM_MEDIA_IMAGE)]]) {
  uint h_pos = cm_group_id(0); // block column
  uint v_pos = cm_group_id(1); // block row

  matrix<uchar, 8, 32> in;  // 8 rows x 32 bytes (2 extra rows, 8 extra bytes of halo)
  matrix<uchar, 6, 24> out;
  matrix<float, 6, 24> m;

  read(ibuf, h_pos * 24, v_pos * 6, in); // X in bytes, Y in rows

  m = in.select<6, 1, 24, 1>(1, 3); // uchar -> float conversion on assignment
  m += in.select<6, 1, 24, 1>(0, 0);
  m += in.select<6, 1, 24, 1>(0, 3);
  m += in.select<6, 1, 24, 1>(0, 6);
  m += in.select<6, 1, 24, 1>(1, 0);
  m += in.select<6, 1, 24, 1>(1, 6);
  m += in.select<6, 1, 24, 1>(2, 0);
  m += in.select<6, 1, 24, 1>(2, 3);
  m += in.select<6, 1, 24, 1>(2, 6);

  out = m * 0.111f; // float -> uchar conversion (truncation) on assignment

  write(obuf, h_pos * 24, v_pos * 6, out);
}
