// Same job three ways: each group of 16 consecutive floats -> its sum and an
// inclusive prefix scan. (1) SYCL sub-group (implicit SIMD), (2) ESIMD.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
namespace esimd = sycl::ext::intel::esimd;

int main() {
  constexpr size_t n = 1 << 20;
  sycl::queue q{sycl::gpu_selector_v};
  float *x = sycl::malloc_shared<float>(n, q);
  float *scan_sg = sycl::malloc_shared<float>(n, q);
  float *sum_sg = sycl::malloc_shared<float>(n / 16, q);
  float *scan_es = sycl::malloc_shared<float>(n, q);
  float *sum_es = sycl::malloc_shared<float>(n / 16, q);
  for (size_t i = 0; i < n; ++i) x[i] = float(i % 7);

  // (1) SIMT + sub-group: one work-item per element; the compiler packs 16
  // work-items into one hardware thread (reqd_sub_group_size(16)); cross-lane
  // work goes through group algorithms.
  q.parallel_for(sycl::nd_range<1>{n, 128}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
     sycl::sub_group sg = it.get_sub_group();
     size_t i = it.get_global_id(0);
     float v = x[i];
     scan_sg[i] = sycl::inclusive_scan_over_group(sg, v, sycl::plus<float>());
     float s = sycl::reduce_over_group(sg, v, sycl::plus<float>());
     if (sg.get_local_id()[0] == 0) sum_sg[i / 16] = s;
   }).wait();

  // (2) ESIMD: one work-item IS the hardware thread and owns all 16 values.
  q.parallel_for(sycl::range<1>{n / 16}, [=](sycl::id<1> t) SYCL_ESIMD_KERNEL {
     esimd::simd<float, 16> v = esimd::block_load<float, 16>(x + t * 16);
     esimd::simd<float, 16> s = v;
     // Hillis-Steele: 4 shifted adds on register regions.
     { esimd::simd<float, 16> sh = 0; sh.select<15, 1>(1) = s.select<15, 1>(0); s += sh; }
     { esimd::simd<float, 16> sh = 0; sh.select<14, 1>(2) = s.select<14, 1>(0); s += sh; }
     { esimd::simd<float, 16> sh = 0; sh.select<12, 1>(4) = s.select<12, 1>(0); s += sh; }
     { esimd::simd<float, 16> sh = 0; sh.select<8, 1>(8) = s.select<8, 1>(0); s += sh; }
     esimd::block_store(scan_es + t * 16, s);
     sum_es[t] = esimd::reduce<float>(v, std::plus<>());
   }).wait();

  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) bad += scan_sg[i] != scan_es[i];
  for (size_t g = 0; g < n / 16; ++g) bad += sum_sg[g] != sum_es[g] || sum_sg[g] != scan_es[g * 16 + 15];
  std::printf("device %s: sub-group vs ESIMD: %zu mismatches -> %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str(), bad, bad ? "FAIL" : "PASS");
  return bad != 0;
}
