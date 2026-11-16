// host_spmv.cpp -- CSR sparse matrix-vector product with spmv_genx.cpp,
// SYCL host, checked against a CPU reference.
//
//   host_spmv spmv_genx.{spv,bin} [nrows]
//
// The matrix mixes row lengths 0..200 (empty rows, very sparse rows, rows
// spanning several 32-wide slices) so every slice width and the tail masking
// are exercised. Values and x are small integers, so every product and sum
// is exact in float and the GPU must match the CPU bit for bit.
// ONEAPI_DEVICE_SELECTOR=level_zero:gpu or opencl:gpu picks the backend.
//
// Build: icpx -fsycl -O2 -fp-model=precise host_spmv.cpp -lze_loader -lOpenCL -o host_spmv
#include "cm_sycl.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

// The kernel reads ACOL/ANZ in slices of up to 32 elements, so the last
// slice runs up to 31 elements past the end. With USM pointers that read is
// not bounds-checked as with an OpenCL buffer: pad both arrays with zeros.
constexpr size_t SLICE = 32;

// Device copy of a host vector, with `pad` zero elements after it. The queue
// is out of order: the copy and the zero fill touch disjoint ranges, and the
// caller waits on the queue before the first launch.
template <typename T>
T *to_device(sycl::queue &q, const std::vector<T> &v, size_t pad = 0) {
  T *d = sycl::malloc_device<T>(v.size() + pad, q);
  if (!d) {
    throw std::runtime_error("device allocation failed");
  }
  q.memcpy(d, v.data(), v.size() * sizeof(T));
  if (pad > 0) {
    q.memset(d + v.size(), 0, pad * sizeof(T));
  }
  return d;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s spmv_genx.{spv,bin} [nrows]\n", argv[0]);
    return 2;
  }
  const size_t nrows = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 100000;
  const size_t ncols = nrows;
  if (nrows == 0) {
    std::fprintf(stderr, "nrows must be positive\n");
    return 2;
  }

  try {
    // Queue on the GPU and the CM program.
    sycl::queue q = cmsycl::make_queue();
    cmsycl::program prog = cmsycl::load_program(q, argv[1]);
    sycl::kernel spmv = prog.kernel("spmv_csr");

    // Random CSR matrix. ACOL stores column + 1 (x is padded with x_pad[0] = 0).
    std::vector<uint32_t> rowptr(nrows + 1, 0);
    std::vector<uint32_t> acol;
    std::vector<float> anz;
    unsigned s = 2024;
    auto rnd = [&s]() {
      s = s * 1664525u + 1013904223u;
      return s >> 8;
    };
    for (size_t r = 0; r < nrows; ++r) {
      unsigned kind = rnd() % 10;
      size_t len = kind == 0 ? 0 : kind < 4 ? rnd() % 8 : kind < 8 ? 8 + rnd() % 24 : 32 + rnd() % 169;
      for (size_t k = 0; k < len; ++k) {
        acol.push_back(uint32_t(rnd() % ncols) + 1);
        anz.push_back(float(int(rnd() % 7) - 3));
      }
      rowptr[r + 1] = uint32_t(acol.size());
    }
    std::vector<float> xpad(ncols + 1);
    xpad[0] = 0.0f; // the padded zero that masked lanes read
    for (size_t j = 1; j <= ncols; ++j) {
      xpad[j] = float(int(rnd() % 9) - 4);
    }
    assert(acol.size() == anz.size() && rowptr[nrows] == acol.size());
    std::printf("CSR: %zu rows, %zu nonzeros (%.1f per row)\n", nrows, anz.size(),
                double(anz.size()) / double(nrows));

    // Device USM (SurfaceIndex buffer_t arguments take USM pointers).
    uint32_t *d_rowptr = to_device(q, rowptr);
    uint32_t *d_acol = to_device(q, acol, SLICE);
    float *d_anz = to_device(q, anz, SLICE);
    float *d_x = to_device(q, xpad);
    float *d_y = sycl::malloc_device<float>(nrows, q);
    if (!d_y) {
      throw std::runtime_error("device allocation failed");
    }
    q.wait();

    // One CM thread per row.
    cmsycl::launch(q, spmv, nrows, 1, d_rowptr, d_acol, d_anz, d_x, d_y).wait(); // warm-up
    double best_ms = 1e30;
    for (int rep = 0; rep < 20; ++rep) {
      sycl::event ev = cmsycl::launch(q, spmv, nrows, 1, d_rowptr, d_acol, d_anz, d_x, d_y);
      ev.wait();
      double ms = cmsycl::elapsed_ms(ev);
      best_ms = ms < best_ms ? ms : best_ms;
    }

    // Results back to the host.
    std::vector<float> y(nrows);
    q.memcpy(y.data(), d_y, nrows * sizeof(float)).wait();
    sycl::free(d_rowptr, q);
    sycl::free(d_acol, q);
    sycl::free(d_anz, q);
    sycl::free(d_x, q);
    sycl::free(d_y, q);

    // Check against the CPU.
    size_t bad = 0;
    for (size_t r = 0; r < nrows; ++r) {
      float ref = 0.0f;
      for (uint32_t k = rowptr[r]; k < rowptr[r + 1]; ++k) {
        ref += anz[k] * xpad[acol[k]];
      }
      if (y[r] != ref) {
        if (bad < 5) {
          std::printf("y[%zu] = %g, expected %g (row length %u)\n", r, y[r], ref, rowptr[r + 1] - rowptr[r]);
        }
        ++bad;
      }
    }
    std::printf("kernel %.3f ms (best of 20)\n", best_ms);
    return cmsycl::report("spmv_csr", nrows, bad) ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
