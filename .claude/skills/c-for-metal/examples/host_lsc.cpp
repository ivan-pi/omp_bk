// host_lsc.cpp -- check and time the LSC kernels of lsc_genx.cpp (SYCL host).
//
//   host_lsc lsc_genx.{spv,bin} [log2 n]
//
// The copies are timed with SYCL profiling events (best of 10) and compared
// with the legacy OWord copy. SurfaceIndex arguments and copy_ptr's svmptr_t
// arguments both take device USM pointers. On a GPU without LSC (before DG2 /
// Xe-HPG) it prints a "skipped:" line and exits 0. A current cmc's .spv needs
// its matched runtime (`cm_run ./host_lsc lsc_genx.spv`); its .bin also runs
// on the system driver. ONEAPI_DEVICE_SELECTOR=level_zero:gpu or opencl:gpu
// picks the backend.
//
// Build: icpx -fsycl -O2 -fp-model=precise host_lsc.cpp -lze_loader -lOpenCL -o host_lsc
#include "cm_sycl.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

constexpr size_t CHUNK = 512; // floats per thread, LSC_CHUNK of the kernel
constexpr size_t LB = 64;     // floats per thread in gather_rev / DWords in hist_lsc
static_assert(CHUNK % LB == 0, "n is checked against CHUNK only");

template <typename T>
T *device_alloc(sycl::queue &q, size_t n) {
  T *d = sycl::malloc_device<T>(n, q);
  if (!d) {
    throw std::runtime_error("device allocation failed");
  }
  return d;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s lsc_genx.{spv,bin} [log2 n]\n", argv[0]);
    return 2;
  }
  const size_t n = size_t(1) << (argc > 2 ? std::atoi(argv[2]) : 26);
  if (n % CHUNK != 0) {
    std::fprintf(stderr, "n must be a multiple of %zu\n", CHUNK);
    return 2;
  }
  try {
    // Queue on the GPU; skip before loading anything on a GPU without LSC.
    sycl::queue q = cmsycl::make_queue();
    if (!cmsycl::has_lsc(q.get_device())) {
      std::printf("skipped: this GPU has no LSC data port (needs DG2 / Xe-HPG or newer)\n");
      return 0;
    }
    cmsycl::program prog = cmsycl::load_program(q, argv[1]);
    bool ok = true;

    // Data: x on the device, y the copy target.
    std::vector<float> x(n);
    std::vector<float> y(n);
    for (size_t i = 0; i < n; ++i) {
      x[i] = float(i % 65521) - 7.5f;
    }
    float *dx = device_alloc<float>(q, n);
    float *dy = device_alloc<float>(q, n);
    q.memcpy(dx, x.data(), n * sizeof(float)).wait();
    const double gb = 2.0 * n * sizeof(float) * 1e-9;

    // Surface copies: legacy OWord, LSC, LSC with cache hints; then the
    // flat-address copy (copy_ptr: svmptr_t arguments, same USM pointers).
    for (const char *name : {"copy_oword", "copy_lsc", "copy_stream", "copy_ptr"}) {
      sycl::kernel k = prog.kernel(name);
      double best = 1e30;
      for (int rep = 0; rep < 10; ++rep) {
        q.fill(dy, 0.0f, n).wait();
        best = std::min(best, cmsycl::elapsed_ms(cmsycl::launch(q, k, n / CHUNK, 1, dx, dy)));
      }
      q.memcpy(y.data(), dy, n * sizeof(float)).wait();
      size_t bad = 0;
      for (size_t i = 0; i < n; ++i) {
        bad += y[i] != x[i];
      }
      char what[96];
      std::snprintf(what, sizeof what, "%s %.3f ms %.0f GB/s", name, best, gb / (best * 1e-3));
      ok &= cmsycl::report(what, n, bad);
    }

    // Gather: reverse.
    {
      sycl::kernel k = prog.kernel("gather_rev");
      cmsycl::launch(q, k, n / LB, 1, dx, dy, uint32_t(n)).wait();
      q.memcpy(y.data(), dy, n * sizeof(float)).wait();
      size_t bad = 0;
      for (size_t i = 0; i < n; ++i) {
        bad += y[i] != x[n - 1 - i];
      }
      ok &= cmsycl::report("gather_rev (SIMD16 gather)", n, bad);
    }
    sycl::free(dx, q);
    sycl::free(dy, q);

    // Atomics: byte histogram of 1M random bytes.
    {
      const size_t words = size_t(1) << 18;
      static_assert((size_t(1) << 18) % LB == 0, "whole threads");
      std::vector<uint32_t> data(words);
      std::vector<uint32_t> bins(256, 0);
      std::vector<uint32_t> ref(256, 0);
      unsigned s = 12345;
      for (uint32_t &w : data) {
        s = s * 1664525u + 1013904223u;
        w = s;
        for (int b = 0; b < 4; ++b) {
          ++ref[(w >> (8 * b)) & 0xff];
        }
      }
      uint32_t *dd = device_alloc<uint32_t>(q, words);
      uint32_t *db = device_alloc<uint32_t>(q, 256);
      q.memcpy(dd, data.data(), words * sizeof(uint32_t));
      q.memset(db, 0, 256 * sizeof(uint32_t));
      q.wait();
      sycl::kernel k = prog.kernel("hist_lsc");
      cmsycl::launch(q, k, words / LB, 1, dd, db).wait();
      q.memcpy(bins.data(), db, 256 * sizeof(uint32_t)).wait();
      sycl::free(dd, q);
      sycl::free(db, q);
      size_t bad = 0;
      for (int i = 0; i < 256; ++i) {
        bad += bins[i] != ref[i];
      }
      ok &= cmsycl::report("hist_lsc (LSC atomic IINC)", 256, bad);
    }
    return ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
