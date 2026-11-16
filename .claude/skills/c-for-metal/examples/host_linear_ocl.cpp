// host_linear_ocl.cpp -- run the linear filter of linear_genx.cpp on OpenCL
// 2D images (CL_R / CL_UNSIGNED_INT8) with the C++ bindings and compare with
// a CPU reference.
//
//   host_linear_ocl linear_genx.{spv,bin}
//
// Build: g++ -std=c++17 -O2 host_linear_ocl.cpp -lOpenCL -o host_linear_ocl
#include "cm_ocl.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s linear_genx.{spv,bin}\n", argv[0]);
    return 2;
  }
  constexpr size_t W = 3 * 640; // bytes per row (640 RGB pixels), multiple of 24
  constexpr size_t H = 480;     // rows, multiple of 6
  static_assert(W % 24 == 0 && H % 6 == 0, "the block grid must tile the image");

  try {
    // Device, context, queue and the CM program.
    cl::Device dev = cmocl::pick_gpu();
    std::printf("device: %s\n", dev.getInfo<CL_DEVICE_NAME>().c_str());
    cl::Context ctx(dev);
    cl::CommandQueue q(ctx, dev);
    cl::Program prog = cmocl::load_program(ctx, dev, argv[1]);

    // Random input image.
    std::vector<unsigned char> in(W * H), out(W * H, 0);
    unsigned s = 7;
    for (size_t i = 0; i < in.size(); ++i) {
      s = s * 1664525u + 1013904223u;
      in[i] = (unsigned char)(s >> 24);
    }

    // [[type("image2d_t")]] (media block) arguments take 2D images. A 1-byte
    // format makes the kernel's X offsets byte offsets AND makes the hardware
    // clamp out-of-range reads per byte (a 4-byte RGBA format would clamp per
    // 4-byte texel instead).
    cl::ImageFormat fmt(CL_R, CL_UNSIGNED_INT8);
    cl::Image2D din(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, fmt, W, H, 0, in.data());
    cl::Image2D dout(ctx, CL_MEM_WRITE_ONLY, fmt, W, H);

    // 2D thread space: one thread per 24x6-byte output block.
    cl::KernelFunctor<cl::Image2D, cl::Image2D> linear(prog, "linear");
    linear(cl::EnqueueArgs(q, cl::NDRange(W / 24, H / 6), cl::NDRange(1, 1)), din, dout).wait();

    std::array<cl::size_type, 3> origin{0, 0, 0}, region{W, H, 1};
    q.enqueueReadImage(dout, CL_TRUE, origin, region, 0, 0, out.data());

    // Same summation order as the kernel: (1,3) first, then the other eight.
    const int dys[9] = {1, 0, 0, 0, 1, 1, 2, 2, 2};
    const int dxs[9] = {3, 0, 3, 6, 0, 6, 0, 3, 6};
    size_t bad = 0;
    for (size_t y = 0; y < H; ++y) {
      for (size_t x = 0; x < W; ++x) {
        float m = 0.0f;
        for (int t = 0; t < 9; ++t) {
          size_t yy = std::min(y + dys[t], H - 1); // hardware clamps to the edge
          size_t xx = std::min(x + dxs[t], W - 1);
          m += float(in[yy * W + xx]);
        }
        unsigned char r = (unsigned char)(m * 0.111f); // float -> uchar truncates
        if (out[y * W + x] != r) {
          if (bad < 5) {
            std::printf("out(%zu,%zu) = %d, expected %d\n", y, x, out[y * W + x], r);
          }
          ++bad;
        }
      }
    }
    return cmocl::report("linear 1920x480 bytes", W * H, bad) ? 0 : 1;
  } catch (const cl::Error &e) {
    std::fprintf(stderr, "OpenCL error %d in %s\n", e.err(), e.what());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
  }
  return 1;
}
