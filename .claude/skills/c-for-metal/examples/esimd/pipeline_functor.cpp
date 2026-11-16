// pipeline_functor.cpp -- ESIMD kernel forms and stage pipelines.
//
//  1. A kernel written as a FUNCTOR STRUCT (operator() marked
//     SYCL_ESIMD_KERNEL) instead of a lambda: handy when kernels are reused,
//     templated or composed. It runs a pipeline of ESIMD_INLINE stage
//     functions on one register-resident chunk -- load once, three stages in
//     registers, store once (the CM way of fusing work per hardware thread).
//  2. A second ESIMD kernel as a LAMBDA, launched after the first in the same
//     queue: kernels form a pipeline through memory (in-order queue).
//
// Stage A: y = 3x - 7   Stage B: clamp to [-50, 50] (merge)   Stage C: inclusive
// prefix sum within the 32-element chunk (log-step shifts with select).
// Kernel 2 gathers each chunk's last prefix value = chunk total.
//
// Build: icpx -fsycl -O2 pipeline_functor.cpp -o pipeline_functor
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cstdio>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
using esimd::simd;

constexpr int VL = 32; // elements per hardware thread (one chunk)

// ---- stages: plain functions on register values -----------------------------

ESIMD_INLINE simd<int, VL> stage_affine(simd<int, VL> x) { return x * 3 - 7; }

ESIMD_INLINE simd<int, VL> stage_clamp(simd<int, VL> y) {
  y.merge(simd<int, VL>(50), y > 50); // predicated moves, no branches
  y.merge(simd<int, VL>(-50), y < -50);
  return y;
}

// Hillis-Steele scan: log2(VL) steps of shifted adds on register regions.
// The region SIZE (VL - S) is a template argument, so S must be a constant
// expression: `#pragma unroll` over a runtime loop counter is not enough
// (select offsets may be runtime values, sizes and strides may not).
template <int S>
ESIMD_INLINE void scan_step(simd<int, VL> &v) {
  if constexpr (S < VL) {
    simd<int, VL> shifted = 0;
    shifted.select<VL - S, 1>(S) = v.select<VL - S, 1>(0);
    v += shifted;
    scan_step<2 * S>(v);
  }
}

ESIMD_INLINE simd<int, VL> stage_prefix(simd<int, VL> v) {
  scan_step<1>(v);
  return v;
}

// ---- kernel 1: functor struct ------------------------------------------------

struct FusedPipeline {
  const int *in;
  int *out;

  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    size_t base = it.get_global_id(0) * VL;  // one work-item == one HW thread

    simd<int, VL> v = esimd::block_load<int, VL>(in + base);

    v = stage_affine(v);
    v = stage_clamp(v);
    v = stage_prefix(v);

    esimd::block_store(out + base, v);
  }
};

int main() {
  constexpr size_t n = 1 << 16;
  constexpr size_t threads = n / VL;

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  int *in = sycl::malloc_shared<int>(n, q);
  int *out = sycl::malloc_shared<int>(n, q);
  int *totals = sycl::malloc_shared<int>(threads, q);
  for (size_t i = 0; i < n; ++i) {
    in[i] = int(i % 41) - 20;
  }

  // Kernel 1: the functor.
  q.parallel_for(sycl::nd_range<1>{threads, 1}, FusedPipeline{in, out});

  // Kernel 2: a lambda that consumes kernel 1's output (in-order queue).
  q.parallel_for(sycl::nd_range<1>{threads / VL, 1}, [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    size_t t0 = it.get_global_id(0) * VL;          // first chunk handled here
    simd<uint32_t, VL> lane(0, 1);                  // 0, 1, ..., VL-1
    simd<uint32_t, VL> byte_off = (lane * VL + (VL - 1)) * sizeof(int); // last of each chunk
    simd<int, VL> last = esimd::gather<int, VL>(out + t0 * VL, byte_off);
    esimd::block_store(totals + t0, last);
  });
  q.wait();

  // CPU reference.
  size_t bad = 0;
  for (size_t t = 0; t < threads; ++t) {
    int run = 0;
    for (int i = 0; i < VL; ++i) {
      int y = in[t * VL + i] * 3 - 7;
      y = y > 50 ? 50 : (y < -50 ? -50 : y);
      run += y;
      bad += out[t * VL + i] != run;
    }
    bad += totals[t] != run;
  }
  std::printf("pipeline (functor + lambda): %zu values checked, %zu mismatches -> %s\n",
              n + threads, bad, bad ? "FAIL" : "PASS");

  sycl::free(in, q);
  sycl::free(out, q);
  sycl::free(totals, q);
  return bad ? 1 : 0;
}
