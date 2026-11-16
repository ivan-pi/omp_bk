// host_emu.cpp -- run a CM kernel on the CPU through the CM SDK's emulator
// ("shim" layer: an OpenCL-C-API look-alike backed by libcm/libigfxcmrt_emu).
//
// No GPU, no cmc, no SPIR-V: the kernel source is compiled by the host g++
// (-DCMRT_EMU -DSHIM) into a shared library ("<kernel>.shim") that exports
// the kernel through EXPORT_SIGNATURE(); clCreateProgramWithBinary takes the
// CONTENTS of that library (it is written to a temp file and dlopen'ed).
// This is a debugging aid: use gdb/printf on kernel code, check semantics.
// The emulator (github.com/intel/cm-cpu-emulation) is archived/discontinued.
//
// The shim implements the OpenCL C API only, so this host uses the C API
// (the GPU hosts of this skill use the C++ bindings). See run_emu.sh.
#include <CL/cl.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#ifndef KERNEL_LIB
#define KERNEL_LIB "vadd.shim"
#endif

int main() {
  constexpr size_t CHUNK = 128; // floats per CM thread, see ../vadd_genx.cpp
  constexpr size_t n = 1 << 14;
  cl_int err = CL_SUCCESS;

  // "Device", context, queue: provided by the shim (CM_RT_PLATFORM picks the emulated GPU).
  cl_platform_id platform;
  cl_device_id dev;
  clGetPlatformIDs(1, &platform, nullptr);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);

  // Program: the bytes of the kernel's shared library.
  std::ifstream f(KERNEL_LIB, std::ios::binary);
  std::vector<unsigned char> lib(std::istreambuf_iterator<char>(f), {});
  if (lib.empty()) {
    std::fprintf(stderr, "cannot read %s\n", KERNEL_LIB);
    return 1;
  }
  size_t len = lib.size();
  const unsigned char *bin = lib.data();
  cl_program prog = clCreateProgramWithBinary(ctx, 1, &dev, &len, &bin, nullptr, &err);
  clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
  cl_kernel k = clCreateKernel(prog, "vadd", &err);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "clCreateKernel: %d (kernel not exported? see run_emu.sh)\n", err);
    return 1;
  }

  // Data and buffers.
  std::vector<float> a(n), b(n), c(n, -1.0f);
  for (size_t i = 0; i < n; ++i) {
    a[i] = float(i % 1000);
    b[i] = 0.5f * float(i % 77);
  }
  cl_mem da = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, n * sizeof(float), a.data(), &err);
  cl_mem db = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, n * sizeof(float), b.data(), &err);
  cl_mem dc = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(float), nullptr, &err);
  clSetKernelArg(k, 0, sizeof(cl_mem), &da);
  clSetKernelArg(k, 1, sizeof(cl_mem), &db);
  clSetKernelArg(k, 2, sizeof(cl_mem), &dc);

  // Same SIMD1 dispatch as on the GPU: one work-item per CM thread.
  size_t global = n / CHUNK;
  size_t local = 1;
  err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
  clEnqueueReadBuffer(q, dc, CL_TRUE, 0, n * sizeof(float), c.data(), 0, nullptr, nullptr);

  // Check.
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    bad += c[i] != a[i] + b[i];
  }
  std::printf("emulated vadd (CPU): %zu checked, %zu mismatches -> %s\n", n, bad,
              bad ? "FAIL" : "PASS");

  clReleaseMemObject(da);
  clReleaseMemObject(db);
  clReleaseMemObject(dc);
  clReleaseKernel(k);
  clReleaseProgram(prog);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);
  return bad ? 1 : 0;
}
