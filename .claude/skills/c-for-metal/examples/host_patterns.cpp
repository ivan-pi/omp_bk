// host_patterns.cpp -- run every kernel of patterns_genx.cpp through SYCL
// (cm_sycl.hpp) and compare with a CPU reference.
//
//   host_patterns patterns_genx.{spv,bin,cpp} [test-name-substring]
//
// SurfaceIndex [[type("buffer_t")]] and svmptr_t parameters both take USM
// pointers; scalars go by value. SIMD1 dispatch: the range counts CM threads,
// the local range counts threads per group (group_sum needs GS = 8).
// Works with the Level Zero and the OpenCL backend
// (ONEAPI_DEVICE_SELECTOR=level_zero:gpu or opencl:gpu).
//
// Build: icpx -fsycl -O2 -fp-model=precise host_patterns.cpp -lze_loader -lOpenCL -o host_patterns
#include "cm_sycl.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

struct Env {
  sycl::queue q;
  cmsycl::program prog;
};

int failures = 0;

void check(const char *name, size_t checked, size_t bad) {
  failures += !cmsycl::report(name, checked, bad);
}

// A shared USM array of n elements, filled with `value`.
template <typename T>
T *shared(Env &e, size_t n, T value) {
  T *p = sycl::malloc_shared<T>(n, e.q);
  if (p == nullptr) {
    throw std::runtime_error("malloc_shared failed");
  }
  for (size_t i = 0; i < n; ++i) {
    p[i] = value;
  }
  return p;
}

// transpose8: 2D thread space of 8x8 tiles. cmsycl::launch2d takes it in CM
// order (x = cm_group_id(0) first) and reverses it for SYCL, whose LAST range
// dimension is the backend's dimension 0.
void test_transpose(Env &e) {
  const uint32_t n = 256;
  static_assert(n % 8 == 0, "transpose8 works on 8x8 tiles");
  float *a = shared<float>(e, n * n, 0.0f);
  float *b = shared<float>(e, n * n, 0.0f);
  for (size_t i = 0; i < size_t(n) * n; ++i) {
    a[i] = float(i);
  }

  // Launch.
  sycl::kernel k = e.prog.kernel("transpose8");
  cmsycl::launch2d(e.q, k, n / 8, n / 8, 1, 1, a, b, n).wait(); // CM order: x = tile column

  // Check.
  size_t bad = 0;
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      bad += b[j * n + i] != a[i * n + j];
    }
  }
  check("transpose8", size_t(n) * n, bad);
  sycl::free(a, e.q);
  sycl::free(b, e.q);
}

void test_hist(Env &e) {
  const size_t n = 1 << 20;
  static_assert(n % 32 == 0, "hist256: 32 bytes per thread");
  unsigned char *d = shared<unsigned char>(e, n, 0);
  uint32_t *h = shared<uint32_t>(e, 256, 0);
  std::vector<uint32_t> ref(256, 0);
  unsigned s = 12345;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1103515245u + 12345u;
    d[i] = (i % 7 == 0) ? 42 : (unsigned char)(s >> 24); // skewed: many hits on bin 42
    ++ref[d[i]];
  }

  // Launch.
  sycl::kernel k = e.prog.kernel("hist256");
  cmsycl::launch(e.q, k, n / 32, 1, d, h).wait();

  // Check.
  size_t bad = 0;
  for (int i = 0; i < 256; ++i) {
    bad += h[i] != ref[i];
  }
  check("hist256", 256, bad);
  sycl::free(d, e.q);
  sycl::free(h, e.q);
}

void test_clamp(Env &e) {
  const size_t n = 1 << 16;
  float *x = shared<float>(e, n, 0.0f);
  float *y1 = shared<float>(e, n, 0.0f);
  float *y2 = shared<float>(e, n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    x[i] = float(int(i % 41) - 15) * 0.75f;
  }

  // Launch.
  sycl::kernel k = e.prog.kernel("clamp_scale");
  cmsycl::launch(e.q, k, n / 32, 1, x, y1, y2).wait();

  // Check.
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    float r = x[i] < 0.0f ? 0.0f : (x[i] > 10.0f ? x[i] * 0.5f : x[i] + 1.0f);
    bad += (y1[i] != r) + (y2[i] != r);
  }
  check("clamp_scale", 2 * n, bad);
  sycl::free(x, e.q);
  sycl::free(y1, e.q);
  sycl::free(y2, e.q);
}

void test_sums(Env &e) {
  const size_t n = 1 << 16;
  const size_t gs = 8; // GS in patterns_genx.cpp: threads per group
  static_assert(n % (32 * gs) == 0, "whole groups of 32-float chunks");
  float *x = shared<float>(e, n, 0.0f);
  float *c = shared<float>(e, n / 32, 0.0f);
  float *g = shared<float>(e, n / 32, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    x[i] = float(i % 13) * 0.25f; // exact in float: sums are order independent
  }

  // Launch: chunk_sum one thread per chunk, group_sum in groups of GS threads (SLM).
  sycl::kernel chunk_sum = e.prog.kernel("chunk_sum");
  sycl::kernel group_sum = e.prog.kernel("group_sum");
  cmsycl::launch(e.q, chunk_sum, n / 32, 1, x, c).wait();
  cmsycl::launch(e.q, group_sum, n / 32, gs, x, g).wait();

  // Check chunk_sum.
  const size_t nc = n / 32;
  size_t bad = 0;
  for (size_t t = 0; t < nc; ++t) {
    float r = 0.0f;
    for (size_t i = 0; i < 32; ++i) {
      r += x[t * 32 + i];
    }
    bad += c[t] != r;
  }
  check("chunk_sum", nc, bad);

  // Check group_sum.
  const size_t ng = n / 32 / gs;
  bad = 0;
  for (size_t gi = 0; gi < ng; ++gi) {
    float r = 0.0f;
    for (size_t i = 0; i < 32 * gs; ++i) {
      r += x[gi * 32 * gs + i];
    }
    bad += g[gi] != r;
  }
  check("group_sum", ng, bad);
  sycl::free(x, e.q);
  sycl::free(c, e.q);
  sycl::free(g, e.q);
}

void test_reverse(Env &e) {
  const size_t n = 1 << 12;
  int32_t *x = shared<int32_t>(e, n, 0);
  int32_t *r = shared<int32_t>(e, n, 0);
  int32_t *p = shared<int32_t>(e, n, 0);
  for (size_t i = 0; i < n; ++i) {
    x[i] = int32_t(i * 3 + 1);
  }

  // Launch.
  sycl::kernel k = e.prog.kernel("reverse32");
  cmsycl::launch(e.q, k, n / 32, 1, x, r, p).wait();

  // Check.
  size_t bad = 0;
  for (size_t t = 0; t < n / 32; ++t) {
    for (size_t i = 0; i < 32; ++i) {
      bad += r[t * 32 + i] != x[t * 32 + 31 - i];
      bad += p[t * 32 + i] != x[t * 32 + (i / 4) * 4];
    }
  }
  check("reverse32", 2 * n, bad);
  sycl::free(x, e.q);
  sycl::free(r, e.q);
  sycl::free(p, e.q);
}

// vadd_svm: svmptr_t parameters take USM pointers, like buffer_t surfaces.
// Unlike coarse-grained OpenCL SVM, shared USM needs no map/unmap.
void test_svm(Env &e) {
  const size_t n = 1 << 16;
  float *a = shared<float>(e, n, 0.0f);
  float *b = shared<float>(e, n, 0.0f);
  float *c = shared<float>(e, n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    a[i] = float(i);
    b[i] = 2.0f;
  }

  // Launch.
  sycl::kernel k = e.prog.kernel("vadd_svm");
  cmsycl::launch(e.q, k, n / 32, 1, a, b, c).wait();

  // Check.
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    bad += c[i] != float(i) + 2.0f;
  }
  check("vadd_svm", n, bad);
  sycl::free(a, e.q);
  sycl::free(b, e.q);
  sycl::free(c, e.q);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s patterns_genx.{spv,bin,cpp} [test]\n", argv[0]);
    return 2;
  }
  // Optional second argument: run only the tests whose name contains it.
  const char *only = argc > 2 ? argv[2] : "";
  try {
    // Queue on the GPU and the CM program.
    sycl::queue q = cmsycl::make_queue();
    Env e{q, cmsycl::load_program(q, argv[1])};

    // Run the selected tests.
    struct {
      const char *name;
      void (*fn)(Env &);
    } tests[] = {{"transpose8", test_transpose}, {"hist256", test_hist},
                 {"clamp_scale", test_clamp},    {"chunk_sum group_sum", test_sums},
                 {"reverse32", test_reverse},    {"vadd_svm", test_svm}};
    for (auto &t : tests) {
      if (std::strstr(t.name, only)) {
        t.fn(e);
      }
    }
  } catch (const sycl::exception &err) {
    std::fprintf(stderr, "SYCL error: %s\n", err.what());
    return 1;
  } catch (const std::exception &err) {
    std::fprintf(stderr, "%s\n", err.what());
    return 1;
  }
  std::printf("%s\n", failures ? "SOME TESTS FAILED" : "ALL PASS");
  return failures ? 1 : 0;
}
