// hello_world_genx.cpp -- kernel for the CM runtime example (hello_world.cpp).
//
// Uses the CM runtime's 2D thread space ("media walker"): every thread gets
// its (x, y) origin from get_thread_origin_x/y() instead of group/local ids.
// Each thread prints a greeting and stores its linear id at index id of the
// output buffer, so the host can verify that every thread ran exactly once.
//
// For the CM runtime no [[type("...")]] annotations are needed: surfaces are
// bound by the runtime (CmBuffer::GetIndex + SetKernelArg).
#include <cm/cm.h>

extern "C" _GENX_MAIN_ void hello_world(int thread_width, SurfaceIndex out) {
  // Thread coordinates in the thread space, linearized row by row.
  uint x = get_thread_origin_x();
  uint y = get_thread_origin_y();
  uint id = x + y * thread_width;

  printf("thread %u (%u, %u): hello from the GPU\n", id, x, y);

  // One-element scattered write: element `id` (units of uint) of `out`.
  // (A vector of 1 rather than the scalar overload, which the 2021 emulation
  // library does not provide.)
  vector<uint, 1> offset = 0;
  vector<uint, 1> value = id;
  write(out, id, offset, value);
}
