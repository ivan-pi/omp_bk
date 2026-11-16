// host_dpas.cpp -- run dpas_hf from dpas_genx.cpp with SYCL and check D = A * B.
//
//   host_dpas dpas_genx.{spv,bin}
//
// Needs a GPU with XMX units (DG2/Arc A-series or newer); the kernel is built
// with cmc -march=dg2 (cm_build.sh -d dg2). Backend:
// ONEAPI_DEVICE_SELECTOR=level_zero:gpu or opencl:gpu.
//
// Build: icpx -fsycl -O2 -fp-model=precise host_dpas.cpp -lze_loader -lOpenCL -o host_dpas
#include "cm_sycl.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

constexpr int M = 8; // must match dpas_genx.cpp
constexpr int N = 8;
constexpr int K = 16;
static_assert(K % 2 == 0, "VNNI packs pairs of rows of B");

static uint16_t half_bits(float x) { return sycl::bit_cast<uint16_t>(sycl::half(x)); }

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s dpas_genx.{spv,bin}\n", argv[0]);
    return 2;
  }

  try {
    // Queue and the CM program.
    sycl::queue q = cmsycl::make_queue(); // needs XMX (DG2 or newer)
    cmsycl::program prog = cmsycl::load_program(q, argv[1]);
    sycl::kernel dpas = prog.kernel("dpas_hf");

    // Small integers: exact in half, products and sums exact in float.
    float a[M][K];
    float b[K][N];
    for (int i = 0; i < M; ++i) {
      for (int kk = 0; kk < K; ++kk) {
        a[i][kk] = float((i + 2 * kk) % 7 - 3);
      }
    }
    for (int kk = 0; kk < K; ++kk) {
      for (int j = 0; j < N; ++j) {
        b[kk][j] = float((3 * kk + j) % 5 - 2);
      }
    }

    // A: row-major halves. B: VNNI -- for each pair of rows (k, k+1) and
    // column n, B(k, n) and B(k+1, n) are adjacent (low half first).
    std::vector<uint16_t> ah(M * K);
    std::vector<uint16_t> bh(K * N);
    for (int i = 0; i < M; ++i) {
      for (int kk = 0; kk < K; ++kk) {
        ah[i * K + kk] = half_bits(a[i][kk]);
      }
    }
    for (int kk = 0; kk < K; ++kk) {
      for (int j = 0; j < N; ++j) {
        bh[(kk / 2) * (2 * N) + j * 2 + kk % 2] = half_bits(b[kk][j]);
      }
    }

    // Upload, launch one CM thread, read back.
    uint16_t *da = sycl::malloc_device<uint16_t>(M * K, q);
    uint16_t *db = sycl::malloc_device<uint16_t>(K * N, q);
    float *dd = sycl::malloc_device<float>(M * N, q);
    q.memcpy(da, ah.data(), M * K * sizeof(uint16_t)).wait();
    q.memcpy(db, bh.data(), K * N * sizeof(uint16_t)).wait();
    cmsycl::launch(q, dpas, 1, 1, da, db, dd).wait(); // one CM thread
    std::vector<float> d(M * N);
    q.memcpy(d.data(), dd, M * N * sizeof(float)).wait();
    sycl::free(da, q);
    sycl::free(db, q);
    sycl::free(dd, q);

    // Check against the CPU product.
    size_t bad = 0;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float r = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
          r += a[i][kk] * b[kk][j];
        }
        if (d[i * N + j] != r) {
          if (bad < 5) {
            std::printf("D(%d,%d) = %g, expected %g\n", i, j, d[i * N + j], r);
          }
          ++bad;
        }
      }
    }
    return cmsycl::report("dpas_hf 8x8x16", M * N, bad) ? 0 : 1;
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
