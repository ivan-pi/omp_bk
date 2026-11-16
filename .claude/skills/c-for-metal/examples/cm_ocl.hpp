// cm_ocl.hpp -- small helpers for launching C for Metal kernels with the
// OpenCL C++ bindings (CLHPP, <CL/opencl.hpp>,
// https://github.khronos.org/OpenCL-CLHPP/).
//
//   cl::Device dev = cmocl::pick_gpu();     // Intel GPU taking SPIR-V; $CM_DEVICE prefers a name
//   cl::Context ctx(dev);
//   cl::CommandQueue q(ctx, dev);
//   cl::Program prog = cmocl::load_program(ctx, dev, "kernel.spv");
//   cl::KernelFunctor<cl::Buffer, cl::Buffer> k(prog, "name");   // extern "C" name
//   k(cl::EnqueueArgs(q, cl::NDRange(threads), cl::NDRange(1)), a, b);
//
// CM kernels use SIMD1 dispatch: every NDRange counts CM hardware THREADS
// (global = number of threads, local = threads per group).
//
// These helpers belong to this skill, not to C for Metal. CM's own host API
// is the CM runtime (CreateCmDevice / CmQueue / CmTask, libigfxcmrt from the
// Intel media driver; the 2021 CM SDK ships only its CPU emulation). With the
// OpenCL / Level Zero / SYCL runtimes a CM kernel is just a SPIR-V module.
#pragma once

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif
#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 210 // clCreateProgramWithIL
#endif
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace cmocl {

inline std::vector<char> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open " + path);
  }
  return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

inline bool ends_with(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// True if the device's OpenCL driver ingests SPIR-V (clCreateProgramWithIL),
// which is all a CM kernel compiled with `cmc -emit-spirv` needs -- plus an
// Intel GPU driver whose compiler has the vector-compute backend (-vc-codegen).
inline bool accepts_spirv(const cl::Device &dev) {
  std::string il = dev.getInfo<CL_DEVICE_IL_VERSION>();
  return il.find("SPIR-V") != std::string::npos;
}

// Pick the GPU to run CM kernels on: an Intel GPU that accepts SPIR-V.
// If `want` (default: $CM_DEVICE) is set, prefer a device whose name contains
// it (e.g. "UHD" to choose an integrated GPU); otherwise take the one with the
// most compute units, usually the discrete GPU when there is one. Not part of any CM API:
// this is plain OpenCL device selection.
inline cl::Device pick_gpu(std::string want = "") {
  if (want.empty()) {
    const char *env = std::getenv("CM_DEVICE");
    want = env ? env : "";
  }

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  cl::Device best;
  int best_score = -1;
  for (const cl::Platform &p : platforms) {
    std::vector<cl::Device> devs;
    try {
      p.getDevices(CL_DEVICE_TYPE_GPU, &devs);
    } catch (const cl::Error &) { // CL_DEVICE_NOT_FOUND on CPU/FPGA platforms
      continue;
    }

    for (const cl::Device &d : devs) {
      bool intel = d.getInfo<CL_DEVICE_VENDOR_ID>() == 0x8086;
      if (!intel || !accepts_spirv(d)) {
        continue;
      }
      bool named = !want.empty() && d.getInfo<CL_DEVICE_NAME>().find(want) != std::string::npos;
      int score = (named ? 1 << 20 : 0) + int(d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
      if (score > best_score) {
        best = d;
        best_score = score;
      }
    }
  }

  if (best_score < 0) {
    throw std::runtime_error("no Intel GPU with SPIR-V support found");
  }
  return best;
}

// Intel GPU IP version (cl_intel_device_attribute_query), packed as
// major << 22 | minor << 14 | revision, e.g. 12.55.8 for DG2; 0 if unknown.
inline unsigned ip_version(const cl::Device &dev) {
  constexpr cl_device_info kIpVersionIntel = 0x4250; // CL_DEVICE_IP_VERSION_INTEL
  cl_uint v = 0;
  if (clGetDeviceInfo(dev(), kIpVersionIntel, sizeof v, &v, nullptr) != CL_SUCCESS) {
    return 0;
  }
  return v;
}

// True for GPUs with the LSC data port: XeHP (12.50) and newer -- DG2, PVC,
// MTL/ARL, Xe2, Xe3. Gen9 .. Gen12LP (TGL, RKL, ADL, DG1) have only the legacy one.
inline bool has_lsc(const cl::Device &dev) {
  unsigned v = ip_version(dev);
  unsigned major = v >> 22;
  unsigned minor = (v >> 14) & 0xff;
  return major > 12 || (major == 12 && minor >= 50);
}

// Build options for CM SPIR-V. "-vc-codegen" selects the vector-compute (CM)
// backend. On LSC GPUs "-ftranslate-legacy-memory-intrinsics" rewrites legacy
// dataport messages (legacy atomics fail at run time on DG2 without it);
// pre-LSC GPUs reject that flag.
inline const char *vc_options(const cl::Device &dev) {
  return has_lsc(dev) ? "-vc-codegen -ftranslate-legacy-memory-intrinsics" : "-vc-codegen";
}

inline void build_or_throw(cl::Program &prog, const cl::Device &dev, const char *opts) {
  try {
    prog.build({dev}, opts);
  } catch (const cl::BuildError &e) {
    std::string log;
    for (const auto &entry : e.getBuildLog()) {
      log += entry.second;
    }
    throw std::runtime_error(std::string("build failed (") + opts + "):\n" + log);
  }
}

// Create and build a program from a CM kernel file:
//   .spv  SPIR-V from `cmc -emit-spirv -fcmocl`  -> clCreateProgramWithIL + "-vc-codegen"
//   .bin  device binary from ocloc              -> clCreateProgramWithBinary
//   .cpp  CM source                             -> clCreateProgramWithSource + "-cmc"
//         (needs a CM front end matching the driver; fails with the 2021 SDK)
inline cl::Program load_program(const cl::Context &ctx, const cl::Device &dev, const std::string &path) {
  std::vector<char> blob = read_file(path);

  if (ends_with(path, ".spv")) {
    cl::Program prog(ctx, blob);
    build_or_throw(prog, dev, vc_options(dev));
    return prog;
  }

  if (ends_with(path, ".bin")) {
    cl::Program::Binaries bins{std::vector<unsigned char>(blob.begin(), blob.end())};
    cl::Program prog(ctx, {dev}, bins);
    build_or_throw(prog, dev, "");
    return prog;
  }

  cl::Program prog(ctx, std::string(blob.begin(), blob.end()));
  build_or_throw(prog, dev, "-cmc");
  return prog;
}

// Print one PASS/FAIL line; returns true on success.
inline bool report(const std::string &what, size_t checked, size_t bad) {
  std::printf("%-28s %9zu checked, %zu mismatches -> %s\n", what.c_str(), checked, bad,
              bad ? "FAIL" : "PASS");
  return bad == 0;
}

} // namespace cmocl
