// host_scan.cpp -- check and time scan_genx.cpp (SYCL).
//
//   host_scan scan_genx.{spv,bin} [log2_ints] [chunk]
//
// chunk must equal the SCAN_CHUNK the kernels were compiled with (default
// 1024); $LB_CHUNK must equal the kernel's LB_CHUNK (default 512). The array
// length is not a multiple of the chunk, so the tail logic is exercised.
// Values are small integers: the CPU result is exact. Timing: best of 20 from
// SYCL profiling events (cmsycl::elapsed_ms); arrays live in device USM.
// Backend: ONEAPI_DEVICE_SELECTOR=level_zero:gpu or opencl:gpu; $CM_DEVICE
// prefers a GPU by name (e.g. UHD).
//
// Build: icpx -fsycl -O2 -fp-model=precise host_scan.cpp -lze_loader -lOpenCL -o host_scan
#include "cm_sycl.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <vector>

namespace {

// Best-of-N GPU time of a sequence of launches (sum of their kernel times).
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

// Mismatches of out against the inclusive or exclusive scan of a.
size_t check_scan(const std::vector<int32_t> &a, const std::vector<int32_t> &out, uint32_t exclusive) {
  size_t bad = 0;
  int32_t x = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (exclusive) {
      bad += out[i] != x; // out[i] = x; x += a[i]
      x += a[i];
    } else {
      x += a[i]; // x += a[i]; out[i] = x
      bad += out[i] != x;
    }
  }
  return bad;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s scan_genx.{spv,bin} [log2_ints] [chunk]\n", argv[0]);
    return 2;
  }
  const int log2_n = argc > 2 ? std::atoi(argv[2]) : 26;
  const size_t chunk = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 1024;
  int failures = 0;

  try {
    // Device, profiling in-order queue and the CM program.
    sycl::queue q = cmsycl::make_queue(); // in-order: passes 1-3 and the resets depend on each other
    cmsycl::program prog = cmsycl::load_program(q, argv[1]);

    // ---- 1. the 16-element demo: a[i] = i --------------------------------
    {
      int32_t *d = sycl::malloc_device<int32_t>(32, q);
      cmsycl::launch(q, prog.kernel("scan16_demo"), 1, 1, d).wait();
      std::vector<int32_t> r(32);
      q.memcpy(r.data(), d, 32 * sizeof(int32_t)).wait();
      sycl::free(d, q);
      size_t bad = 0;
      for (int i = 0; i < 16; ++i) {
        bad += r[i] != i * (i + 1) / 2;      // inclusive: 0, 1, 3, 6, ...
        bad += r[16 + i] != i * (i - 1) / 2; // exclusive: 0, 0, 1, 3, ...
      }
      std::printf("inclusive: ");
      for (int i = 0; i < 16; ++i) {
        std::printf("%d ", r[i]);
      }
      std::printf("\nexclusive: ");
      for (int i = 0; i < 16; ++i) {
        std::printf("%d ", r[16 + i]);
      }
      std::printf("\n");
      failures += !cmsycl::report("scan16_demo", 32, bad);
    }

    // ---- 2. full-array scans ---------------------------------------------
    const size_t n = (size_t(1) << log2_n) - 13;
    const size_t nchunks = (n + chunk - 1) / chunk;
    std::vector<int32_t> a(n);
    unsigned s = 99;
    for (size_t i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      a[i] = int32_t(s >> 29) - 3; // -3..4
    }

    // Device arrays and kernels.
    int32_t *da = sycl::malloc_device<int32_t>(n, q);
    int32_t *dsums = sycl::malloc_device<int32_t>(nchunks + 512, q);
    int32_t *doffs = sycl::malloc_device<int32_t>(nchunks + 512, q);
    int32_t *dout = sycl::malloc_device<int32_t>(n, q);
    q.memcpy(da, a.data(), n * sizeof(int32_t)).wait();
    sycl::kernel p1 = prog.kernel("scan_p1_reduce");   // (a, sums)
    sycl::kernel p2 = prog.kernel("scan_p2_offsets");  // (sums, offs, nchunks)
    sycl::kernel p3 = prog.kernel("scan_p3_scan");     // (a, offs, out, exclusive)
    sycl::kernel copy = prog.kernel("copy_i32");       // (a, out)

    auto scan = [&](uint32_t exclusive) {
      return std::vector<sycl::event>{cmsycl::launch(q, p1, nchunks, 1, da, dsums),
                                      cmsycl::launch(q, p2, 1, 1, dsums, doffs, uint32_t(nchunks)),
                                      cmsycl::launch(q, p3, nchunks, 1, da, doffs, dout, exclusive)};
    };

    // Timing: the copy reference, each pass, the whole scan.
    double t_copy = best_ms(20, [&] { return std::vector<sycl::event>{cmsycl::launch(q, copy, nchunks, 1, da, dout)}; });
    double t_p1 = best_ms(20, [&] { return std::vector<sycl::event>{cmsycl::launch(q, p1, nchunks, 1, da, dsums)}; });
    double t_p2 = best_ms(20, [&] {
      return std::vector<sycl::event>{cmsycl::launch(q, p2, 1, 1, dsums, doffs, uint32_t(nchunks))};
    });
    double t_p3 = best_ms(20, [&] {
      return std::vector<sycl::event>{cmsycl::launch(q, p3, nchunks, 1, da, doffs, dout, 0u)};
    });
    double t_inc = best_ms(20, [&] { return scan(0u); });
    double t_exc = best_ms(20, [&] { return scan(1u); });

    double gb = double(n * sizeof(int32_t)) / 1e9;
    std::printf("n=%zu ints, %zu chunks of %zu\n", n, nchunks, chunk);
    std::printf("copy (read+write) %.3f ms (%.0f GB/s moved)\n", t_copy, 2 * gb / (t_copy * 1e-3));
    std::printf("pass1 reduce %.3f ms (%.0f GB/s) | pass2 offsets %.3f ms | pass3 scan %.3f ms (%.0f GB/s moved)\n",
                t_p1, gb / (t_p1 * 1e-3), t_p2, t_p3, 2 * gb / (t_p3 * 1e-3));
    std::printf("inclusive scan total %.3f ms, exclusive %.3f ms (%.0f GB/s moved; %.2f x copy time)\n", t_inc,
                t_exc, 3 * gb / (t_inc * 1e-3), t_inc / t_copy);

    // Check both scans.
    std::vector<int32_t> out(n);
    for (uint32_t exclusive : {0u, 1u}) {
      for (sycl::event &ev : scan(exclusive)) {
        ev.wait();
      }
      q.memcpy(out.data(), dout, n * sizeof(int32_t)).wait();
      size_t bad = check_scan(a, out, exclusive);
      failures += !cmsycl::report(exclusive ? "exclusive scan" : "inclusive scan", n, bad);
    }

    // ---- 3. single-pass decoupled look-back scan -------------------------
    {
      const char *lb_env = std::getenv("LB_CHUNK");
      const size_t lb = lb_env ? std::strtoul(lb_env, nullptr, 10) : 512; // LB_CHUNK in scan_genx.cpp
      assert(lb > 0);
      const size_t lchunks = (n + lb - 1) / lb;
      const size_t stride = lchunks + 16;
      const size_t state_words = 16 + 3 * stride; // [0] ticket, [1..2] error report, FLAG/AGG/INC from 16
      uint32_t *dstate = sycl::malloc_device<uint32_t>(state_words, q);
      sycl::kernel lookback = prog.kernel("scan_lookback"); // (a, out, state, stride, exclusive)

      // The ticket counter and flags must be zero before every run.
      auto run = [&](uint32_t exclusive) {
        sycl::event fill = q.fill(dstate, 0u, state_words);
        sycl::event ev = cmsycl::launch(q, lookback, lchunks, 1, da, dout, dstate, uint32_t(stride), exclusive);
        return std::vector<sycl::event>{fill, ev};
      };
      double t_lb_inc = best_ms(20, [&] { return run(0u); });
      double t_lb_exc = best_ms(20, [&] { return run(1u); });
      double t_kernel = best_ms(20, [&] {
        q.fill(dstate, 0u, state_words);
        return std::vector<sycl::event>{
            cmsycl::launch(q, lookback, lchunks, 1, da, dout, dstate, uint32_t(stride), 0u)};
      });
      std::printf("look-back scan (%zu chunks of %zu): kernel %.3f ms; with state reset: inclusive %.3f ms, "
                  "exclusive %.3f ms (%.0f GB/s moved; %.2f x copy time)\n",
                  lchunks, lb, t_kernel, t_lb_inc, t_lb_exc, 2 * gb / (t_lb_inc * 1e-3), t_lb_inc / t_copy);

      // Check, several runs each: races would show up as flakiness.
      for (uint32_t exclusive : {0u, 1u}) {
        for (int rep = 0; rep < 5; ++rep) {
          for (sycl::event &ev : run(exclusive)) {
            ev.wait();
          }
          q.memcpy(out.data(), dout, n * sizeof(int32_t)).wait();
          uint32_t err[2];
          q.memcpy(err, dstate + 1, sizeof err).wait();
          if (err[0]) { // a chunk gave up waiting (LB_MAX_SPIN): report instead of hanging
            std::printf("look-back: chunk %u gave up waiting for chunk %u\n", err[0] - 1, err[1] - 1);
          }
          size_t bad = (err[0] ? 1 : 0) + check_scan(a, out, exclusive);
          if (bad || rep == 4) {
            failures += !cmsycl::report(exclusive ? "look-back exclusive scan" : "look-back inclusive scan", n, bad);
            if (bad) {
              break;
            }
          }
        }
      }
      sycl::free(dstate, q);
    }
    sycl::free(da, q);
    sycl::free(dsums, q);
    sycl::free(doffs, q);
    sycl::free(dout, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return failures ? 1 : 0;
}
