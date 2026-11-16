// host_sycl.cpp -- launch a CM kernel from SYCL (the skill's default host).
//
//   host_sycl vadd_genx.{spv,bin,cpp} [n]
//
// cmsycl::load (cm_sycl.hpp) takes SPIR-V from cmc (kernel_compiler +
// "-vc-codegen"), a device binary (Level Zero / OpenCL interop) or CM source
// ("-cmc", current cmc + matched IGC only: run under cm_run). Works with the
// Level Zero and the OpenCL backend (ONEAPI_DEVICE_SELECTOR=level_zero:gpu or
// opencl:gpu). Runs vadd of vadd_genx.cpp and checks it.
//
// Build: icpx -fsycl -O2 -fp-model=precise host_sycl.cpp -lze_loader -lOpenCL -o host_sycl
#include "cm_sycl.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s vadd_genx.{spv,bin,cpp} [n]\n", argv[0]);
    return 2;
  }
  constexpr size_t CHUNK = 128; // floats per CM thread, see vadd_genx.cpp
  const size_t n = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : (1u << 20);
  if (n % CHUNK != 0) {
    std::fprintf(stderr, "n must be a multiple of %zu\n", CHUNK);
    return 2;
  }

  try {
    // Queue on the GPU and the CM program.
    sycl::queue q = cmsycl::make_queue();
    cmsycl::program prog = cmsycl::load_program(q, argv[1]);
    sycl::kernel vadd = prog.kernel("vadd"); // extern "C" kernel name

    // Data in USM: SurfaceIndex [[type("buffer_t")]] parameters take USM pointers.
    float *a = sycl::malloc_shared<float>(n, q);
    float *b = sycl::malloc_shared<float>(n, q);
    float *c = sycl::malloc_shared<float>(n, q);
    for (size_t i = 0; i < n; ++i) {
      a[i] = float(i % 1000);
      b[i] = 0.5f * float(i % 77);
      c[i] = -1.0f;
    }

    // SIMD1 dispatch: one work-item per CM thread.
    sycl::event ev = cmsycl::launch(q, vadd, n / CHUNK, 1, a, b, c);
    ev.wait();

    // Check.
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
      bad += c[i] != a[i] + b[i];
    }
    std::printf("%.3f ms\n", cmsycl::elapsed_ms(ev));
    bool ok = cmsycl::report(std::string(argv[1]) + " vadd", n, bad);
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    return ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
