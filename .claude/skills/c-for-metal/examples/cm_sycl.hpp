// cm_sycl.hpp -- small SYCL helpers for hosting C for Metal kernels (the
// default host API of this skill; cm_ocl.hpp is the OpenCL C++ counterpart).
// These belong to the skill, not to CM: a CM kernel is just a SPIR-V module or
// a device binary, so any Intel GPU whose driver takes SPIR-V can run it.
//
//   sycl::queue q = cmsycl::make_queue();                 // GPU, IN-ORDER, profiling on
//   auto prog = cmsycl::load_program(q, "kernel.spv");    // .spv / .bin / .cpp
//   sycl::kernel k = prog.kernel("vadd");                 // extern "C" name
//   float *a = cmsycl::to_device(q, host_a);              // USM copy (+ optional zero pad)
//   cmsycl::launch(q, k, n_threads, 1, a, b, c);          // SIMD1: range counts threads
//
// Argument mapping: SurfaceIndex [[type("buffer_t")]] and svmptr_t take USM
// pointers; scalars by value. Image surfaces (image2d_t, media block I/O) and
// SPIR-V specialization constants by id are not reachable from SYCL's
// kernel_compiler: use the OpenCL host (cm_ocl.hpp) for those kernels.
//
// Two differences from an OpenCL host that bite CM kernels:
//   * USM has no bounds checks. A cl_mem surface returns 0 for out-of-range
//     reads and drops out-of-range writes, and many CM kernels rely on that
//     for their last, partial block or thread. With USM, pad the allocation
//     to whole blocks/threads (to_device(q, v, pad)).
//   * SYCL queues are out-of-order by default and USM carries no
//     dependencies, so make_queue() returns an IN-ORDER queue, like OpenCL's.
//
// Build: icpx -fsycl -O2 -fp-model=precise host.cpp -lze_loader -lOpenCL -o host
// (icpx defaults to -fp-model=fast, which changes CPU reference results.)
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <CL/cl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace cmsycl {

namespace syclex = sycl::ext::oneapi::experimental;

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

// IP version word (major << 22 | minor << 14 | revision), as cmc's
// __CM_INTEL_TARGET_ID and OpenCL's CL_DEVICE_IP_VERSION_INTEL encode it.
// SYCL's architecture enum for Intel GPUs is that word shifted left by 8.
inline unsigned ip_version(const sycl::device &dev) {
  auto arch = dev.get_info<syclex::info::device::architecture>();
  return unsigned((uint64_t(arch) >> 8) & 0xffffffffu);
}

struct ip_triple {
  unsigned major;
  unsigned minor;
  unsigned revision;
};

inline ip_triple decode_ip(unsigned v) { return {v >> 22, (v >> 14) & 0xff, v & 0x3f}; }

// LSC data port: Xe-HPG (DG2, 12.55) and newer.
inline bool has_lsc(const sycl::device &dev) {
  unsigned v = ip_version(dev);
  unsigned major = v >> 22;
  unsigned minor = (v >> 14) & 0xff;
  return major > 12 || (major == 12 && minor >= 55);
}

// Backend options for CM SPIR-V: the vector-compute backend, plus the
// legacy-message translation on LSC GPUs (pre-LSC GPUs reject that flag).
inline const char *vc_options(const sycl::device &dev) {
  return has_lsc(dev) ? "-vc-codegen -ftranslate-legacy-memory-intrinsics" : "-vc-codegen";
}

// An Intel GPU, preferring one whose name contains `want` (default: $CM_DEVICE),
// then the one with the most compute units. ONEAPI_DEVICE_SELECTOR picks the backend.
inline sycl::device pick_gpu(std::string want = "") {
  if (want.empty() && std::getenv("CM_DEVICE")) {
    want = std::getenv("CM_DEVICE");
  }
  sycl::device best;
  long best_score = -1;
  for (const sycl::device &d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    if (d.get_info<sycl::info::device::vendor_id>() != 0x8086) {
      continue;
    }
    long score = d.get_info<sycl::info::device::max_compute_units>();
    if (!want.empty() && d.get_info<sycl::info::device::name>().find(want) != std::string::npos) {
      score += 1L << 20;
    }
    if (score > best_score) {
      best_score = score;
      best = d;
    }
  }
  if (best_score < 0) {
    throw std::runtime_error("no Intel GPU found");
  }
  return best;
}

// In-order by default: launches, copies and fills on USM run in submission
// order, as on an OpenCL queue. Pass in_order = false to manage events yourself.
inline sycl::queue make_queue(std::string want = "", bool in_order = true) {
  sycl::property_list props = in_order ? sycl::property_list{sycl::property::queue::enable_profiling{},
                                                             sycl::property::queue::in_order{}}
                                       : sycl::property_list{sycl::property::queue::enable_profiling{}};
  sycl::queue q(pick_gpu(want), props);
  std::printf("device: %s (%s)\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
              q.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "opencl");
  return q;
}

// Native binary (from ocloc / cmc -o) -> executable bundle, through the backend's interop.
inline sycl::kernel_bundle<sycl::bundle_state::executable> load_binary(const sycl::queue &q,
                                                                        const std::vector<char> &bin) {
  sycl::context ctx = q.get_context();
  sycl::device dev = q.get_device();
  if (q.get_backend() == sycl::backend::ext_oneapi_level_zero) {
    auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = bin.size();
    desc.pInputModule = reinterpret_cast<const uint8_t *>(bin.data());
    ze_module_handle_t module = nullptr;
    if (zeModuleCreate(zctx, zdev, &desc, &module, nullptr) != ZE_RESULT_SUCCESS) {
      throw std::runtime_error("zeModuleCreate (native) failed: binary not built for this GPU?");
    }
    return sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero, sycl::bundle_state::executable>(
        {module, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx);
  }
  cl_context cctx = sycl::get_native<sycl::backend::opencl>(ctx);
  cl_device_id cdev = sycl::get_native<sycl::backend::opencl>(dev);
  const unsigned char *data = reinterpret_cast<const unsigned char *>(bin.data());
  size_t size = bin.size();
  cl_int status = 0;
  cl_int err = 0;
  cl_program prog = clCreateProgramWithBinary(cctx, 1, &cdev, &size, &data, &status, &err);
  if (err != CL_SUCCESS || clBuildProgram(prog, 1, &cdev, "", nullptr, nullptr) != CL_SUCCESS) {
    throw std::runtime_error("OpenCL binary load failed: binary not built for this GPU?");
  }
  auto exe = sycl::make_kernel_bundle<sycl::backend::opencl, sycl::bundle_state::executable>(prog, ctx);
  clReleaseProgram(prog);
  clReleaseContext(cctx);
  clReleaseDevice(cdev);
  return exe;
}

// Load a CM program:
//   .spv  SPIR-V from cmc               -> kernel_compiler (spirv) + "-vc-codegen"
//   .bin  device binary (ocloc, cmc -o) -> Level Zero / OpenCL interop
//   .cpp  CM source                     -> kernel_compiler ("opencl" source) + "-cmc"
//         (needs the intel-igc-cm front end and its matched IGC: run under cm_run)
// Do not gate on ext_oneapi_can_compile(spirv): it wrongly returns false.
inline sycl::kernel_bundle<sycl::bundle_state::executable> load(const sycl::queue &q, const std::string &path,
                                                                std::string extra_options = "") {
  std::vector<char> blob = read_file(path);
  try {
    if (ends_with(path, ".bin")) {
      return load_binary(q, blob);
    }
    if (ends_with(path, ".cpp")) {
      auto src = syclex::create_kernel_bundle_from_source(q.get_context(), syclex::source_language::opencl,
                                                          std::string(blob.begin(), blob.end()));
      return syclex::build(src, syclex::properties{syclex::build_options{"-cmc " + extra_options}});
    }
    std::vector<std::byte> spv(blob.size());
    std::memcpy(spv.data(), blob.data(), blob.size());
    auto src = syclex::create_kernel_bundle_from_source(q.get_context(), syclex::source_language::spirv, spv);
    std::string opts = std::string(vc_options(q.get_device())) + " " + extra_options;
    return syclex::build(src, syclex::properties{syclex::build_options{opts}});
  } catch (const sycl::exception &e) {
    throw std::runtime_error("loading " + path + " failed: " + e.what());
  }
}

// A loaded CM program. kernel(name) works for every source: bundles built by
// kernel_compiler answer ext_oneapi_get_kernel; interop bundles from a .bin
// need a backend kernel object (zeKernelCreate / clCreateKernel) wrapped by
// sycl::make_kernel.
struct program {
  sycl::queue q;
  sycl::kernel_bundle<sycl::bundle_state::executable> bundle;
  bool interop = false;

  sycl::kernel kernel(const std::string &name) {
    if (!interop) {
      return bundle.ext_oneapi_get_kernel(name);
    }
    sycl::context ctx = q.get_context();
    if (q.get_backend() == sycl::backend::ext_oneapi_level_zero) {
      auto modules = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(bundle);
      ze_kernel_desc_t desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
      desc.pKernelName = name.c_str();
      ze_kernel_handle_t k = nullptr;
      if (zeKernelCreate(modules.at(0), &desc, &k) != ZE_RESULT_SUCCESS) {
        throw std::runtime_error("no kernel " + name + " in the binary");
      }
      return sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
          {bundle, k, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx);
    }
    auto progs = sycl::get_native<sycl::backend::opencl>(bundle);
    cl_int err = 0;
    cl_kernel k = clCreateKernel(progs.at(0), name.c_str(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("no kernel " + name + " in the binary");
    }
    sycl::kernel kern = sycl::make_kernel<sycl::backend::opencl>(k, ctx);
    clReleaseKernel(k);
    for (cl_program p : progs) {
      clReleaseProgram(p);
    }
    return kern;
  }
};

inline program load_program(const sycl::queue &q, const std::string &path, std::string extra_options = "") {
  return program{q, load(q, path, extra_options), ends_with(path, ".bin")};
}

// SIMD1 dispatch: `threads` CM threads in groups of `group` threads.
template <typename... Args>
sycl::event launch(sycl::queue &q, const sycl::kernel &k, size_t threads, size_t group, Args... args) {
  return q.submit([&](sycl::handler &h) {
    h.set_args(args...);
    h.parallel_for(sycl::nd_range<1>{threads, group}, k);
  });
}

// 2D thread space in CM order: (x, y) = (cm_group_id(0)/cm_local_id(0) side,
// dimension 1 side). SYCL's LAST range dimension maps to the backend's
// dimension 0, so the ranges are reversed here; don't pass a CM-order
// nd_range<2> to parallel_for directly.
template <typename... Args>
sycl::event launch2d(sycl::queue &q, const sycl::kernel &k, size_t threads_x, size_t threads_y,
                     size_t group_x, size_t group_y, Args... args) {
  return q.submit([&](sycl::handler &h) {
    h.set_args(args...);
    h.parallel_for(sycl::nd_range<2>{{threads_y, threads_x}, {group_y, group_x}}, k);
  });
}

// USM allocation that throws instead of returning null.
template <typename T>
T *alloc_device(sycl::queue &q, size_t n) {
  T *p = sycl::malloc_device<T>(n, q);
  if (p == nullptr) {
    throw std::runtime_error("sycl::malloc_device failed");
  }
  return p;
}

template <typename T>
T *alloc_shared(sycl::queue &q, size_t n) {
  T *p = sycl::malloc_shared<T>(n, q);
  if (p == nullptr) {
    throw std::runtime_error("sycl::malloc_shared failed");
  }
  return p;
}

// Device copy of a host vector, followed by `pad` zero elements (see the
// bounds-check note at the top). Waits, so the host vector may change after.
template <typename T>
T *to_device(sycl::queue &q, const std::vector<T> &v, size_t pad = 0) {
  T *p = alloc_device<T>(q, v.size() + pad);
  if (!v.empty()) {
    q.memcpy(p, v.data(), v.size() * sizeof(T));
  }
  if (pad > 0) {
    q.memset(p + v.size(), 0, pad * sizeof(T));
  }
  q.wait();
  return p;
}

template <typename T>
void to_host(sycl::queue &q, const T *p, std::vector<T> &v) {
  q.memcpy(v.data(), p, v.size() * sizeof(T)).wait();
}

// Kernel time of a finished or pending event (waits first).
inline double elapsed_ms(sycl::event e) {
  e.wait();
  return 1e-6 * double(e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                       e.get_profiling_info<sycl::info::event_profiling::command_start>());
}

// One PASS/FAIL line; returns true on success.
inline bool report(const std::string &what, size_t checked, size_t bad) {
  std::printf("%-28s %9zu checked, %zu mismatches -> %s\n", what.c_str(), checked, bad, bad ? "FAIL" : "PASS");
  return bad == 0;
}

} // namespace cmsycl
