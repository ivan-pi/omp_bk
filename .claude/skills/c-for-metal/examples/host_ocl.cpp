// host_ocl.cpp -- launch a CM kernel with the OpenCL C++ bindings (CLHPP).
//
//   host_ocl <kernel.spv|kernel.bin|kernel.cpp> [n]
//
// The program is created from SPIR-V (IL + "-vc-codegen"), a device binary,
// or CM source ("-cmc", needs a matching CM front end) -- see
// cmocl::load_program in cm_ocl.hpp. Runs vadd of vadd_genx.cpp and checks it.
//
// Build: g++ -std=c++17 -O2 host_ocl.cpp -lOpenCL -o host_ocl
#include "cm_ocl.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s kernel.{spv,bin,cpp} [n]\n", argv[0]);
    return 2;
  }
  constexpr size_t CHUNK = 128; // floats per CM thread, see vadd_genx.cpp
  const size_t n = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : (1u << 20);
  if (n % CHUNK != 0) {
    std::fprintf(stderr, "n must be a multiple of %zu\n", CHUNK);
    return 2;
  }

  try {
    // Device, context, queue and the CM program (.spv / .bin / .cpp).
    cl::Device dev = cmocl::pick_gpu();
    std::printf("device: %s\n", dev.getInfo<CL_DEVICE_NAME>().c_str());
    cl::Context ctx(dev);
    cl::CommandQueue q(ctx, dev);
    cl::Program prog = cmocl::load_program(ctx, dev, argv[1]);

    // Host data.
    std::vector<float> a(n), b(n), c(n, -1.0f);
    for (size_t i = 0; i < n; ++i) {
      a[i] = float(i % 1000);
      b[i] = 0.5f * float(i % 77);
    }

    // SurfaceIndex [[type("buffer_t")]] parameters take ordinary buffers.
    cl::Buffer da(ctx, a.begin(), a.end(), true); // read-only, copied from host
    cl::Buffer db(ctx, b.begin(), b.end(), true);
    cl::Buffer dc(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float));

    // SIMD1 dispatch: one work-item == one CM hardware thread.
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> vadd(prog, "vadd"); // extern "C" name
    vadd(cl::EnqueueArgs(q, cl::NDRange(n / CHUNK), cl::NDRange(1)), da, db, dc).wait();

    // Check against the CPU.
    cl::copy(q, dc, c.begin(), c.end());
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
      bad += c[i] != a[i] + b[i];
    }
    return cmocl::report(std::string(argv[1]) + " vadd", n, bad) ? 0 : 1;
  } catch (const cl::Error &e) {
    std::fprintf(stderr, "OpenCL error %d in %s\n", e.err(), e.what());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
