// cm_sycl.hpp -- SYCL host helpers for the C for Metal kernels of cm/ (the
// hosts bk1_cm.cpp and bk5_cm.cpp).  A CM kernel is a device binary from
// ocloc (cm/Makefile); SYCL loads it through the backend's interop:
//
//   sycl::queue q = cmsycl::make_queue();                    // in order, profiling on
//   cmsycl::program prog = cmsycl::load_program(q, path);    // kernel_binary("bk1", p, ept)
//   sycl::kernel k = prog.kernel("name");                    // extern "C" name
//   cmsycl::launch(q, k, threads, group, a, b, n);           // SIMD1: ranges count threads
//
// SurfaceIndex [[type("buffer_t")]] parameters take USM pointers, scalars go
// by value.  Backend: ONEAPI_DEVICE_SELECTOR (level_zero:gpu, the default, or
// opencl:gpu); $CM_DEVICE prefers a GPU whose name contains it.
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <CL/cl.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "bk_common.h"
#include "bk_harness.h"   // fail_mark
#include "bk_sycl.h"

namespace cmsycl {

// --------------------------------------------------------------------------
// device, queue and program
// --------------------------------------------------------------------------

// An Intel GPU, preferring one whose name contains $CM_DEVICE, then the one
// with the most compute units (the discrete GPU)
inline sycl::device pick_gpu()
{
    const std::string want = bk::get_env("CM_DEVICE").value_or("");
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

// the binary cm/Makefile writes for `kernel` (bk1, bk5) at order p with ept
// elements per thread: <dir>/<kernel>_cm_<p>[_e<ept>].bin, dir $BK_CM_DIR or
// the executable's directory
inline std::string kernel_binary(const std::string& kernel, const int p, const int ept)
{
    const std::string dir = bk::get_env("BK_CM_DIR").value_or(exe_dir());
    return dir + "/" + kernel + "_cm_" + std::to_string(p) + (ept == 1 ? "" : "_e" + std::to_string(ept)) + ".bin";
}

// A device binary loaded through Level Zero or OpenCL interop.  Interop
// bundles have no ext_oneapi_get_kernel: kernel() creates the backend kernel
// and wraps it.
struct program {
    sycl::queue q;
    sycl::kernel_bundle<sycl::bundle_state::executable> bundle;

    sycl::kernel kernel(const std::string& name)
    {
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

inline program load_program(const sycl::queue& q, const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::vector<char> bin(std::istreambuf_iterator<char>(f), {});
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
            throw std::runtime_error("zeModuleCreate failed for " + path + ": binary not built for this GPU?");
        }
        return program{q, sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero, sycl::bundle_state::executable>(
                              {module, sycl::ext::oneapi::level_zero::ownership::transfer}, ctx)};
    }
    cl_context cctx = sycl::get_native<sycl::backend::opencl>(ctx);
    cl_device_id cdev = sycl::get_native<sycl::backend::opencl>(dev);
    const unsigned char* data = reinterpret_cast<const unsigned char*>(bin.data());
    const size_t size = bin.size();
    cl_int status = 0;
    cl_int err = 0;
    cl_program prog = clCreateProgramWithBinary(cctx, 1, &cdev, &size, &data, &status, &err);
    if (err != CL_SUCCESS || clBuildProgram(prog, 1, &cdev, "", nullptr, nullptr) != CL_SUCCESS) {
        throw std::runtime_error("clCreateProgramWithBinary failed for " + path + ": binary not built for this GPU?");
    }
    auto exe = sycl::make_kernel_bundle<sycl::backend::opencl, sycl::bundle_state::executable>(prog, ctx);
    clReleaseProgram(prog);
    clReleaseContext(cctx);
    clReleaseDevice(cdev);
    return program{q, exe};
}

// --------------------------------------------------------------------------
// data and launch
// --------------------------------------------------------------------------

// The kernels read whole OWords: up to 3 floats past the last element
constexpr std::size_t OVERREAD = 4;

// device copy of v followed by OVERREAD zeros; enqueued only, the caller
// waits (and keeps v alive until then)
template <typename T>
T* to_device(sycl::queue& q, const std::vector<T>& v)
{
    T* p = sycl::malloc_device<T>(v.size() + OVERREAD, q);
    if (p == nullptr) {
        throw std::runtime_error("sycl::malloc_device failed");
    }
    q.memcpy(p, v.data(), v.size() * sizeof(T));
    q.memset(p + v.size(), 0, OVERREAD * sizeof(T));
    return p;
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

// --------------------------------------------------------------------------
// benchmark
// --------------------------------------------------------------------------

// work-group sizes (threads per group) to run: $BK_WG or `dflt`
inline std::vector<int> work_groups(const std::string& dflt)
{
    std::istringstream ws(bk::get_env("BK_WG").value_or(dflt));
    std::vector<int> wgs;
    for (int wg = 0; ws >> wg;) {
        wgs.push_back(wg);
    }
    return wgs;
}

// bandwidth of copy_cm (cm_common.h, COPY_CHUNK floats per thread), the
// reference of the "% of copy_cm" figure: it copies about as many bytes as
// the kernel moves; prints and returns its device GB/s
inline double copy_bandwidth(sycl::queue& q, program& prog, const double bytes, const int ntests)
{
    constexpr std::size_t chunk = 256;   // COPY_CHUNK of cm_common.h
    constexpr std::size_t wg = 64;
    const std::size_t n = std::size_t(bytes / sizeof(float) / 2) / (chunk * wg) * (chunk * wg);
    float* src = sycl::malloc_device<float>(n, q);
    float* dst = sycl::malloc_device<float>(n, q);
    q.fill(src, 1.0f, n).wait();
    const sycl::kernel copy_cm = prog.kernel("copy_cm");
    const bk::launch_times t = bk::time_launches(ntests, [&] { return launch(q, copy_cm, n / chunk, wg, src, dst); });
    sycl::free(src, q);
    sycl::free(dst, q);
    const double moved = 2.0 * n * sizeof(float);
    std::printf("%-14s GB/s = %6.1f  (device: GB/s = %6.1f)\n", "copy_cm", 1e-9 * moved / t.host, 1e-9 * moved / t.device);
    return 1e-9 * moved / t.device;
}

// one result line: host and device GDoF/s and GB/s, the device GB/s as a
// share of copy_cm's, the error against the reference
inline void report(const std::string& name, const double dofs, const double bytes, const bk::launch_times& t,
                   const double copy_gbs, const double relerr)
{
    std::printf("%-14s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f = %3.0f%% of copy_cm)"
                "  relerr = %.1e%s\n",
                name.c_str(), 1e-9 * dofs / t.host, 1e-9 * bytes / t.host, 1e-9 * dofs / t.device,
                1e-9 * bytes / t.device, 100.0 * 1e-9 * bytes / t.device / copy_gbs, relerr, bk::fail_mark(relerr));
}

} // namespace cmsycl
