// host_argmax.cpp -- benchmark and check argmax_genx.cpp (int32 and double)
// against a CPU reference, next to a read-only bandwidth reference kernel (SYCL).
//
//   host_argmax argmax_genx.{spv,bin} [log2_bytes] [chunk]
//
// chunk must equal the ARGMAX_CHUNK the kernel was compiled with (default
// 512). Data: random values with the maximum planted twice (the first
// occurrence must win) and many ties; doubles include negatives, -0.0, +0.0.
// Timing: best of 20 for stage 1 alone, and for the whole argmax (stage 1 +
// every stage-2 launch of the reduction tree), from SYCL profiling events
// (cmsycl::elapsed_ms). Arrays live in device USM, uploaded with q.memcpy.
// Backend: ONEAPI_DEVICE_SELECTOR=level_zero:gpu or opencl:gpu; $CM_DEVICE
// prefers a GPU by name (e.g. UHD).
//
// Build: icpx -fsycl -O2 -fp-model=precise host_argmax.cpp -lze_loader -lOpenCL -o host_argmax
#include "cm_sycl.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <utility>
#include <vector>

namespace {

constexpr size_t P_CHUNK = 1024; // must match argmax_genx.cpp

// Best-of-N GPU time of a sequence of launches (sum of its kernels' times).
double best_ms(int reps, const std::function<std::vector<sycl::event>()> &run) {
  for (sycl::event &ev : run()) {
    ev.wait();
  }
  double best = 1e30;
  for (int i = 0; i < reps; ++i) {
    double ms = 0.0;
    for (sycl::event &ev : run()) {
      ev.wait();
      ms += cmsycl::elapsed_ms(ev);
    }
    best = ms < best ? ms : best;
  }
  return best;
}

template <typename T>
size_t cpu_argmax(const std::vector<T> &a) {
  size_t at = 0;
  for (size_t i = 1; i < a.size(); ++i) {
    if (a[i] > a[at]) { // first occurrence; -0.0 == +0.0 as in IEEE
      at = i;
    }
  }
  return at;
}

// Runs stage 1 and the stage-2 tree; returns the final index and the events.
struct Argmax {
  sycl::queue &q;
  sycl::kernel p1; // (data, part, n, stride)
  sycl::kernel p2; // (in, out, nparts, in_stride, out_stride)
  int arrays;      // partial arrays: 2 for int32 (value, index), 3 for double (khi, klo, index)
  size_t chunk;

  std::vector<sycl::event> run(const void *data, size_t n, uint32_t *ping, uint32_t *pong, size_t stride,
                               uint32_t *index) {
    std::vector<sycl::event> evs;
    size_t parts = (n + chunk - 1) / chunk;
    assert(parts <= stride);
    evs.push_back(cmsycl::launch(q, p1, parts, 1, data, ping, uint32_t(n), uint32_t(stride)));
    uint32_t *in = ping;
    uint32_t *out = pong;
    while (parts > 1) {
      size_t threads = (parts + P_CHUNK - 1) / P_CHUNK;
      evs.push_back(cmsycl::launch(q, p2, threads, 1, in, out, uint32_t(parts), uint32_t(stride),
                                   uint32_t(stride)));
      parts = threads;
      std::swap(in, out);
    }
    if (index) {
      q.memcpy(index, in + (arrays - 1) * stride, sizeof(uint32_t)).wait();
    }
    return evs;
  }
};

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s argmax_genx.{spv,bin} [log2_bytes] [chunk]\n", argv[0]);
    return 2;
  }
  const int log2_bytes = argc > 2 ? std::atoi(argv[2]) : 28; // 256 MiB per array
  const size_t chunk = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 512;
  const size_t bytes = size_t(1) << log2_bytes;
  int failures = 0;

  try {
    // Device, profiling in-order queue and the CM program.
    sycl::queue q = cmsycl::make_queue(); // in-order: the launches depend on each other via USM
    cmsycl::program prog = cmsycl::load_program(q, argv[1]);
    Argmax i32{q, prog.kernel("argmax_i32_p1"), prog.kernel("argmax_i32_p2"), 2, chunk};
    Argmax f64{q, prog.kernel("argmax_f64_p1"), prog.kernel("argmax_f64_p2"), 3, chunk};
    sycl::kernel readsum = prog.kernel("readsum_i32");

    unsigned s = 12345;
    auto rnd = [&s]() {
      s = s * 1664525u + 1013904223u;
      return s;
    };

    // ---- int32 ------------------------------------------------------------
    {
      const size_t n = bytes / sizeof(int32_t) - 37; // not a multiple of the chunk
      std::vector<int32_t> a(n);
      for (size_t i = 0; i < n; ++i) {
        a[i] = int32_t(rnd() >> 4) - (1 << 27); // many repeats, negatives
      }
      a[n / 3] = (1 << 28) + 5; // the maximum, planted twice
      a[n - 2] = (1 << 28) + 5;

      // Device arrays: the data and two partial buffers (ping-pong).
      size_t stride = (n + chunk - 1) / chunk + 256;
      int32_t *da = sycl::malloc_device<int32_t>(n, q);
      uint32_t *ping = sycl::malloc_device<uint32_t>(3 * stride, q);
      uint32_t *pong = sycl::malloc_device<uint32_t>(3 * stride, q);
      q.memcpy(da, a.data(), n * sizeof(int32_t)).wait();
      size_t parts = (n + chunk - 1) / chunk;

      // Timing: the read-only reference, stage 1, the whole argmax.
      double t_sum = best_ms(20, [&] {
        return std::vector<sycl::event>{
            cmsycl::launch(q, readsum, parts, 1, da, ping, uint32_t(n), uint32_t(stride))};
      });
      double t_p1 = best_ms(20, [&] {
        return std::vector<sycl::event>{cmsycl::launch(q, i32.p1, parts, 1, da, ping, uint32_t(n), uint32_t(stride))};
      });
      double t_all = best_ms(20, [&] { return i32.run(da, n, ping, pong, stride, nullptr); });
      uint32_t index = 0;
      i32.run(da, n, ping, pong, stride, &index);

      // Check against the CPU.
      size_t ref = cpu_argmax(a);
      double gb = double(n * sizeof(int32_t)) / 1e9;
      std::printf("int32  n=%zu: read+sum %.3f ms (%.0f GB/s) | stage 1 %.3f ms (%.0f GB/s) | "
                  "argmax total %.3f ms (%.0f GB/s)\n",
                  n, t_sum, gb / (t_sum * 1e-3), t_p1, gb / (t_p1 * 1e-3), t_all, gb / (t_all * 1e-3));
      std::printf("        gpu index %u, cpu index %zu (value %d)\n", index, ref, a[ref]);
      failures += !cmsycl::report("argmax int32", n, index != ref);
      sycl::free(da, q);
      sycl::free(ping, q);
      sycl::free(pong, q);
    }

    // ---- double -----------------------------------------------------------
    {
      const size_t n = bytes / sizeof(double) - 19;
      std::vector<double> a(n);
      for (size_t i = 0; i < n; ++i) {
        a[i] = (double(int32_t(rnd())) / 65536.0) * ((i % 5 == 0) ? -1.0 : 1.0);
      }
      for (size_t i = 0; i < n; i += 1000) {
        a[i] = (i % 2000 == 0) ? -0.0 : 0.0; // signed zeros
      }
      a[n / 4] = 1e300; // the maximum, planted twice
      a[n - 3] = 1e300;

      // Device arrays: the data and two partial buffers (ping-pong).
      size_t stride = (n + chunk - 1) / chunk + 256;
      double *da = sycl::malloc_device<double>(n, q);
      uint32_t *ping = sycl::malloc_device<uint32_t>(3 * stride, q);
      uint32_t *pong = sycl::malloc_device<uint32_t>(3 * stride, q);
      q.memcpy(da, a.data(), n * sizeof(double)).wait();
      size_t parts = (n + chunk - 1) / chunk;

      // Timing: stage 1, the whole argmax.
      double t_p1 = best_ms(20, [&] {
        return std::vector<sycl::event>{cmsycl::launch(q, f64.p1, parts, 1, da, ping, uint32_t(n), uint32_t(stride))};
      });
      double t_all = best_ms(20, [&] { return f64.run(da, n, ping, pong, stride, nullptr); });
      uint32_t index = 0;
      f64.run(da, n, ping, pong, stride, &index);

      // Check against the CPU.
      size_t ref = cpu_argmax(a);
      double gb = double(n * sizeof(double)) / 1e9;
      std::printf("double n=%zu: stage 1 %.3f ms (%.0f GB/s) | argmax total %.3f ms (%.0f GB/s)\n", n, t_p1,
                  gb / (t_p1 * 1e-3), t_all, gb / (t_all * 1e-3));
      std::printf("        gpu index %u, cpu index %zu (value %g)\n", index, ref, a[ref]);
      failures += !cmsycl::report("argmax double", n, index != ref);

      // All-negative data with -0.0/+0.0 ties at the top: the first zero wins.
      std::vector<double> b(1 << 20, -1.0);
      b[777] = -0.0;
      b[5000] = 0.0;
      assert((b.size() + chunk - 1) / chunk <= stride);
      double *db = sycl::malloc_device<double>(b.size(), q);
      q.memcpy(db, b.data(), b.size() * sizeof(double)).wait();
      f64.run(db, b.size(), ping, pong, stride, &index);
      failures += !cmsycl::report("argmax double, signed-zero tie", b.size(), index != 777);
      sycl::free(db, q);
      sycl::free(da, q);
      sycl::free(ping, q);
      sycl::free(pong, q);
    }
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return failures ? 1 : 0;
}
