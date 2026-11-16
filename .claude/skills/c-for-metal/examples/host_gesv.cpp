// host_gesv.cpp -- batched GESV with gesv_genx.cpp, checked against a CPU
// sgetrf/sgetrs reference and by the backward error in double (SYCL host).
//
//   host_gesv gesv_genx.{spv,bin} [batch]
//   host_gesv gesvb_genx.{spv,bin} [batch] lanes [L]   (batch-in-lanes kernel)
//
// Compile the host with the same -DGESV_N / -DGESV_NRHS as the kernel (default
// 16, 4; gesvb_genx.cpp defaults to 4, 1). A is random in [-1, 1] (not
// diagonally dominant, so pivoting matters). The arrays live in device USM;
// they are padded to whole CM threads, because gesv_lanes' last thread reads
// and writes the full L systems and USM pointers are not bounds-checked.
// ONEAPI_DEVICE_SELECTOR=level_zero:gpu or opencl:gpu picks the backend.
// icpx defaults to -fp-model=fast, which shifts the printed "max rel diff"
// of the CPU comparison (not the pivots, the backward error or PASS/FAIL);
// the build line's -fp-model=precise gives the same numbers as a g++ build.
//
// Build: icpx -fsycl -O2 -fp-model=precise host_gesv.cpp -lze_loader -lOpenCL -o host_gesv
#include "cm_sycl.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

#ifndef GESV_N
#define GESV_N 16 // must match the kernel's GESV_N / GESV_NRHS
#endif
#ifndef GESV_NRHS
#define GESV_NRHS 4
#endif
constexpr int N = GESV_N;
constexpr int NR = GESV_NRHS;
static_assert(N >= 1 && N <= 32 && NR >= 1, "the kernels solve N <= 32 with at least one rhs");

// LAPACK-style sgetrf + sgetrs on one column-major system, with the same
// operation order as the kernel (fused multiply-adds, reciprocal pivots).
void cpu_gesv(float *a, float *b, int *ipiv) {
  for (int k = 0; k < N; ++k) {
    int p = k;
    for (int i = k + 1; i < N; ++i) {
      if (std::fabs(a[k * N + i]) > std::fabs(a[k * N + p])) {
        p = i; // first maximum
      }
    }
    ipiv[k] = p + 1;
    if (p != k) {
      for (int j = 0; j < N; ++j) {
        std::swap(a[j * N + k], a[j * N + p]);
      }
      for (int r = 0; r < NR; ++r) {
        std::swap(b[r * N + k], b[r * N + p]);
      }
    }
    float rinv = 1.0f / a[k * N + k];
    for (int i = k + 1; i < N; ++i) {
      a[k * N + i] *= rinv;
    }
    for (int j = k + 1; j < N; ++j) {
      for (int i = k + 1; i < N; ++i) {
        a[j * N + i] = std::fma(-a[j * N + k], a[k * N + i], a[j * N + i]);
      }
    }
    for (int r = 0; r < NR; ++r) {
      for (int i = k + 1; i < N; ++i) {
        b[r * N + i] = std::fma(-b[r * N + k], a[k * N + i], b[r * N + i]);
      }
    }
  }
  for (int k = N - 1; k >= 0; --k) {
    float rinv = 1.0f / a[k * N + k];
    for (int r = 0; r < NR; ++r) {
      b[r * N + k] *= rinv;
      for (int i = 0; i < k; ++i) {
        b[r * N + i] = std::fma(-b[r * N + k], a[k * N + i], b[r * N + i]);
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s gesv_genx.{spv,bin} [batch]\n"
                         "       %s gesvb_genx.{spv,bin} [batch] lanes [L]   (batch-in-lanes kernel)\n",
                 argv[0], argv[0]);
    return 2;
  }
  const size_t batch = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 131072;
  // "lanes": gesvb_genx.cpp's gesv_lanes, L systems per CM thread (GESVB_L).
  const bool lanes = argc > 3 && std::string(argv[3]) == "lanes";
  const size_t per_thread = lanes ? (argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 16) : 1;
  if (batch == 0 || per_thread == 0) {
    std::fprintf(stderr, "batch and L must be positive\n");
    return 2;
  }
  const size_t threads = (batch + per_thread - 1) / per_thread;
  const size_t padded = threads * per_thread; // systems the kernel touches
  assert(padded >= batch && padded - batch < per_thread);

  try {
    // Queue on the GPU and the CM program.
    sycl::queue q = cmsycl::make_queue();
    cmsycl::program prog = cmsycl::load_program(q, argv[1]);
    sycl::kernel gesv = prog.kernel(lanes ? "gesv_lanes" : "gesv_batched");

    // Random systems.
    std::vector<float> a(batch * N * N);
    std::vector<float> b(batch * N * NR);
    unsigned s = 4242;
    auto uni = [&s]() {
      s = s * 1664525u + 1013904223u;
      return float(int(s >> 8) - (1 << 23)) / float(1 << 23); // [-1, 1)
    };
    for (float &v : a) {
      v = uni();
    }
    for (float &v : b) {
      v = uni();
    }

    // Device USM, padded to whole threads (the padding is zeroed once and never checked).
    float *da = sycl::malloc_device<float>(padded * N * N, q);
    float *db = sycl::malloc_device<float>(padded * N * NR, q);
    int32_t *dp = sycl::malloc_device<int32_t>(padded * N, q);
    if (!da || !db || !dp) {
      throw std::runtime_error("device allocation failed");
    }
    q.memset(da, 0, padded * N * N * sizeof(float));
    q.memset(db, 0, padded * N * NR * sizeof(float));
    q.memset(dp, 0, padded * N * sizeof(int32_t));
    q.wait();

    // Timing: the kernel works in place, so re-upload before each run.
    double best = 1e30;
    for (int rep = 0; rep < 10; ++rep) {
      q.memcpy(da, a.data(), a.size() * sizeof(float));
      q.memcpy(db, b.data(), b.size() * sizeof(float));
      q.wait();
      sycl::event ev = cmsycl::launch(q, gesv, threads, 1, da, db, dp, uint32_t(batch));
      ev.wait();
      best = std::min(best, cmsycl::elapsed_ms(ev));
    }

    // Results back to the host.
    std::vector<float> lu(a.size());
    std::vector<float> x(b.size());
    std::vector<int32_t> piv(batch * N);
    q.memcpy(lu.data(), da, lu.size() * sizeof(float));
    q.memcpy(x.data(), db, x.size() * sizeof(float));
    q.memcpy(piv.data(), dp, piv.size() * sizeof(int32_t));
    q.wait();
    sycl::free(da, q);
    sycl::free(db, q);
    sycl::free(dp, q);

    // Checks: pivots and factors against the CPU reference, backward error in double.
    size_t piv_diff = 0;
    double max_lu = 0.0;
    double max_x = 0.0;
    double max_berr = 0.0;
    std::vector<float> ra(N * N);
    std::vector<float> rb(N * NR);
    std::vector<int> rp(N);
    for (size_t t = 0; t < batch; ++t) {
      std::copy_n(&a[t * N * N], N * N, ra.begin());
      std::copy_n(&b[t * N * NR], N * NR, rb.begin());
      cpu_gesv(ra.data(), rb.data(), rp.data());
      bool same = std::equal(rp.begin(), rp.end(), &piv[t * N]);
      piv_diff += !same;
      if (same) {
        for (int i = 0; i < N * N; ++i) {
          max_lu = std::max(max_lu, double(std::fabs(lu[t * N * N + i] - ra[i])) / (1.0 + std::fabs(ra[i])));
        }
        for (int i = 0; i < N * NR; ++i) {
          max_x = std::max(max_x, double(std::fabs(x[t * N * NR + i] - rb[i])) / (1.0 + std::fabs(rb[i])));
        }
      }

      // Backward error ||A x - b||_inf / (||A||_inf ||x||_inf) per right-hand side.
      const float *at = &a[t * N * N];
      double anorm = 0.0;
      for (int i = 0; i < N; ++i) {
        double row = 0.0;
        for (int j = 0; j < N; ++j) {
          row += std::fabs(at[j * N + i]);
        }
        anorm = std::max(anorm, row);
      }
      for (int r = 0; r < NR; ++r) {
        const float *xr = &x[t * N * NR + r * N];
        const float *br = &b[t * N * NR + r * N];
        double res = 0.0;
        double xn = 0.0;
        for (int i = 0; i < N; ++i) {
          double acc = -double(br[i]);
          for (int j = 0; j < N; ++j) {
            acc += double(at[j * N + i]) * double(xr[j]);
          }
          res = std::max(res, std::fabs(acc));
          xn = std::max(xn, std::fabs(double(xr[i])));
        }
        max_berr = std::max(max_berr, res / (anorm * xn));
      }
    }

    // Report.
    double flops = double(batch) * (2.0 / 3.0 * N * N * N + 2.0 * N * N * NR);
    double bytes = double(batch) * (2.0 * (N * N + N * NR) * 4 + N * 4);
    std::printf("%s %zu systems of %dx%d, %d rhs: %.3f ms, %.0f GFLOP/s, %.0f GB/s moved\n",
                lanes ? "gesv_lanes" : "gesv", batch, N, N, NR, best, flops / (best * 1e6), bytes / (best * 1e6));
    std::printf("pivots differ from CPU in %zu systems; max rel diff LU %.2e, X %.2e; max backward error %.2e\n",
                piv_diff, max_lu, max_x, max_berr);
    bool ok = max_berr < 1e-5 && piv_diff * 1000 <= batch; // near-ties may pick another pivot
    return cmsycl::report("batched gesv (backward error, pivots)", batch, ok ? 0 : 1) ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
