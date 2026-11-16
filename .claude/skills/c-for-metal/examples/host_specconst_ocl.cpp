// host_specconst_ocl.cpp -- set the specialization constants of
// specconst_genx.cpp from OpenCL and check the result.
//
//   host_specconst_ocl specconst_genx.spv [blocks scale]
//   host_specconst_ocl specconst_genx.bin  blocks scale   (values baked in by ocloc -spec_const)
//
// For .spv the program is specialized twice: once with the defaults
// (blocks = 4, scale = 0) and once with the given values (default 8, 2.5).
//
// The .spv path needs the IGC matched to the cmc: run as `cm_run ./host_specconst_ocl k.spv`
// (scripts/install_cm_toolchain.sh). ocloc 25.18 has no -spec_const, so a .bin carries the defaults.
//
// Build: g++ -std=c++17 -O2 host_specconst_ocl.cpp -lOpenCL -o host_specconst_ocl
#include "cm_ocl.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

constexpr size_t BLOCK = 32;
constexpr size_t THREADS = 4096;

size_t run(cl::Context &ctx, cl::CommandQueue &q, cl::Program &prog, int blocks, float s) {
  const size_t n = THREADS * blocks * BLOCK;
  std::vector<float> x(n), y(n + BLOCK, -1.0f);
  for (size_t i = 0; i < n; ++i) {
    x[i] = float(i % 977);
  }
  cl::Buffer dx(ctx, x.begin(), x.end(), true);
  cl::Buffer dy(ctx, y.begin(), y.end(), false);
  cl::KernelFunctor<cl::Buffer, cl::Buffer> scale(prog, "scale");
  scale(cl::EnqueueArgs(q, cl::NDRange(THREADS), cl::NDRange(1)), dx, dy).wait();
  cl::copy(q, dy, y.begin(), y.end());
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    bad += y[i] != s * x[i];
  }
  for (size_t i = n; i < n + BLOCK; ++i) {
    bad += y[i] != -1.0f; // nothing written past BLOCKS*32 per thread
  }
  char what[64];
  std::snprintf(what, sizeof what, "spec const blocks=%d s=%g", blocks, s);
  cmocl::report(what, n, bad);
  return bad;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s specconst_genx.{spv,bin} [blocks scale]\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  const int blocks = argc > 2 ? std::atoi(argv[2]) : 8;
  const float s = argc > 3 ? float(std::atof(argv[3])) : 2.5f;
  try {
    cl::Device dev = cmocl::pick_gpu();
    std::printf("device: %s\n", dev.getInfo<CL_DEVICE_NAME>().c_str());
    cl::Context ctx(dev);
    cl::CommandQueue q(ctx, dev);
    size_t bad = 0;
    if (cmocl::ends_with(path, ".bin")) {
      cl::Program prog = cmocl::load_program(ctx, dev, path); // already specialized
      bad += run(ctx, q, prog, blocks, s);
    } else {
      std::vector<char> il = cmocl::read_file(path);
      cl::Program defaults(ctx, il);
      cmocl::build_or_throw(defaults, dev, cmocl::vc_options(dev));
      bad += run(ctx, q, defaults, 4, 0.0f);

      cl::Program prog(ctx, il);
      prog.setSpecializationConstant(0, blocks); // ids from cm_spec_constant<T, ID>
      prog.setSpecializationConstant(1, s);
      cmocl::build_or_throw(prog, dev, cmocl::vc_options(dev));
      bad += run(ctx, q, prog, blocks, s);
    }
    return bad ? 1 : 0;
  } catch (const cl::Error &e) {
    std::fprintf(stderr, "OpenCL error %d in %s\n", e.err(), e.what());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
