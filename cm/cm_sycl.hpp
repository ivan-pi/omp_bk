// cm_sycl.hpp -- SYCL host helpers for the C for Metal kernels of cm/ (the
// hosts bk1_cm.cpp and bk5_cm.cpp).  A CM kernel is a device binary from
// ocloc (or SPIR-V from cmc); SYCL loads it through the backend's interop:
//
//   sycl::queue q = cmsycl::make_queue();                    // in order, profiling on
//   cmsycl::program prog = cmsycl::load_program(q, "k.bin"); // .bin or .spv
//   sycl::kernel k = prog.kernel("name");                    // extern "C" name
//   cmsycl::launch(q, k, threads, group, a, b, n);           // SIMD1: ranges count threads
//
// SurfaceIndex [[type("buffer_t")]] parameters take USM pointers, scalars go
// by value.  Unlike an OpenCL buffer, USM has no bounds checks: a read past
// the allocation does not return 0, so the hosts pad every array the kernels
// may read past its end with zeros.
//
// Backend: ONEAPI_DEVICE_SELECTOR (level_zero:gpu, the default, or
// opencl:gpu).  $CM_DEVICE prefers a GPU whose name contains it.
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace cmsycl {

namespace syclex = sycl::ext::oneapi::experimental;

inline std::vector<char> read_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

inline bool ends_with(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// An Intel GPU, preferring one whose name contains $CM_DEVICE, then the one
// with the most compute units (the discrete GPU)
inline sycl::device pick_gpu()
{
    const char* env = std::getenv("CM_DEVICE");
    const std::string want = env ? env : "";
    sycl::device best;
    long best_score = -1;
    for (const sycl::device& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
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

// In order (USM carries no dependencies), with event profiling
inline sycl::queue make_queue()
{
    return sycl::queue(pick_gpu(), sycl::property_list{sycl::property::queue::enable_profiling{},
                                                       sycl::property::queue::in_order{}});
}

inline const char* backend_name(const sycl::queue& q)
{
    return q.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "opencl";
}

// LSC data port (DG2 and newer): the legacy memory messages of the 2021 cmc
// need -ftranslate-legacy-memory-intrinsics there; older GPUs reject it
inline bool has_lsc(const sycl::device& dev)
{
    const auto arch = dev.get_info<syclex::info::device::architecture>();
    const unsigned v = unsigned((uint64_t(arch) >> 8) & 0xffffffffu);   // major << 22 | minor << 14 | rev
    const unsigned major = v >> 22;
    const unsigned minor = (v >> 14) & 0xff;
    return major > 12 || (major == 12 && minor >= 55);
}

// device binary -> executable bundle, through Level Zero or OpenCL interop
inline sycl::kernel_bundle<sycl::bundle_state::executable> load_binary(const sycl::queue& q,
                                                                        const std::vector<char>& bin)
{
    const sycl::context ctx = q.get_context();
    const sycl::device dev = q.get_device();
    if (q.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
        auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
        ze_module_desc_t desc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
        desc.format = ZE_MODULE_FORMAT_NATIVE;
        desc.inputSize = bin.size();
        desc.pInputModule = reinterpret_cast<const uint8_t*>(bin.data());
        ze_module_handle_t module = nullptr;
        if (zeModuleCreate(zctx, zdev, &desc, &module, nullptr) != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("zeModuleCreate failed: binary not built for this GPU?");
        }
        return sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero, sycl::bundle_state::executable>(
            {module, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx);
    }
    cl_context cctx = sycl::get_native<sycl::backend::opencl>(ctx);
    cl_device_id cdev = sycl::get_native<sycl::backend::opencl>(dev);
    const unsigned char* data = reinterpret_cast<const unsigned char*>(bin.data());
    const size_t size = bin.size();
    cl_int status = 0;
    cl_int err = 0;
    cl_program prog = clCreateProgramWithBinary(cctx, 1, &cdev, &size, &data, &status, &err);
    if (err != CL_SUCCESS || clBuildProgram(prog, 1, &cdev, "", nullptr, nullptr) != CL_SUCCESS) {
        throw std::runtime_error("clCreateProgramWithBinary failed: binary not built for this GPU?");
    }
    auto exe = sycl::make_kernel_bundle<sycl::backend::opencl, sycl::bundle_state::executable>(prog, ctx);
    clReleaseProgram(prog);
    clReleaseContext(cctx);
    clReleaseDevice(cdev);
    return exe;
}

// A loaded CM program: .bin (ocloc) through interop, or .spv (cmc
// -emit-spirv) through kernel_compiler with the vector-compute backend
// (`options` adds backend options, e.g. -double-GRF)
struct program {
    sycl::queue q;
    sycl::kernel_bundle<sycl::bundle_state::executable> bundle;
    bool interop;

    // interop bundles have no ext_oneapi_get_kernel: create the backend
    // kernel and wrap it
    sycl::kernel kernel(const std::string& name)
    {
        if (!interop) {
            return bundle.ext_oneapi_get_kernel(name);
        }
        const sycl::context ctx = q.get_context();
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

inline program load_program(const sycl::queue& q, const std::string& path, const std::string& options = "")
{
    const std::vector<char> blob = read_file(path);
    try {
        if (ends_with(path, ".bin")) {
            return program{q, load_binary(q, blob), true};
        }
        std::vector<std::byte> spv(blob.size());
        std::memcpy(spv.data(), blob.data(), blob.size());
        auto src = syclex::create_kernel_bundle_from_source(q.get_context(), syclex::source_language::spirv, spv);
        const std::string opts = std::string("-vc-codegen") +
                                 (has_lsc(q.get_device()) ? " -ftranslate-legacy-memory-intrinsics" : "") + " " + options;
        return program{q, syclex::build(src, syclex::properties{syclex::build_options{opts}}), false};
    } catch (const sycl::exception& e) {
        throw std::runtime_error("loading " + path + " failed: " + e.what());
    }
}

// SIMD1 dispatch: `threads` CM threads in work-groups of `group` threads
template <typename... Args>
sycl::event launch(sycl::queue& q, const sycl::kernel& k, const size_t threads, const size_t group, Args... args)
{
    return q.submit([&](sycl::handler& h) {
        h.set_args(args...);
        h.parallel_for(sycl::nd_range<1>{threads, group}, k);
    });
}

// device copy of v followed by `pad` zeros
template <typename T>
T* to_device(sycl::queue& q, const std::vector<T>& v, const size_t pad = 0)
{
    T* p = sycl::malloc_device<T>(v.size() + pad, q);
    if (p == nullptr) {
        throw std::runtime_error("sycl::malloc_device failed");
    }
    q.memcpy(p, v.data(), v.size() * sizeof(T));
    if (pad > 0) {
        q.memset(p + v.size(), 0, pad * sizeof(T));
    }
    q.wait();
    return p;
}

// --------------------------------------------------------------------------
// benchmark helpers of the hosts
// --------------------------------------------------------------------------

struct times {
    double host;     // submit to completion
    double device;   // event profiling
};

// one warm-up launch, then the best host and device time of ntests launches
template <typename F>
times time_launches(const int ntests, F&& launch)
{
    launch().wait();
    times best{std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    for (int t = 0; t < ntests; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        sycl::event ev = launch();
        ev.wait();
        const auto t1 = std::chrono::steady_clock::now();
        best.host = std::min(best.host, std::chrono::duration<double>(t1 - t0).count());
        const double ns = double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                 ev.get_profiling_info<sycl::info::event_profiling::command_start>());
        best.device = std::min(best.device, 1e-9 * ns);
    }
    return best;
}

// directory of the running executable (the default place of the kernels)
inline std::string exe_dir()
{
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    const std::string s(buf, std::size_t(n));
    return s.substr(0, s.rfind('/'));
}

inline size_t round_up_to(const size_t n, const size_t m) { return (n + m - 1) / m * m; }

} // namespace cmsycl
