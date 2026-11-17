// bk_sycl.h -- queue and timing of the SYCL harnesses (sycl/, tinytc/):
// one warm-up launch (timed: it includes any JIT compilation), then the best
// host time (submit to completion) and the best device time (event
// profiling) over ntests launches.
//
// Kernel functors are launched themselves, never through a wrapper functor
// holding the kernel as a member: a wrapper changes IGC's code generation
// (sycl/README.md, convention 2).  Required sub-group and work-group sizes go
// in the functor's get(syclex::properties_tag), from launch_properties.

#ifndef BK_SYCL_H
#define BK_SYCL_H

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <limits>

#include "bk_common.h"

namespace bk {

// The launch requirements of a kernel functor, for its
//   auto get(sycl::ext::oneapi::experimental::properties_tag) const
// sg: the required sub-group size, wg: the required (1D) work-group size;
// 0 leaves either to the compiler and the launch.
template <int sg, int wg = 0>
auto launch_properties()
{
    namespace syclex = sycl::ext::oneapi::experimental;
    if constexpr (sg > 0 && wg > 0) {
        return syclex::properties{syclex::sub_group_size<sg>, syclex::work_group_size<wg>};
    } else if constexpr (sg > 0) {
        return syclex::properties{syclex::sub_group_size<sg>};
    } else if constexpr (wg > 0) {
        return syclex::properties{syclex::work_group_size<wg>};
    } else {
        return syclex::properties{};
    }
}

// The launch geometry of a kernel: elements per work-group (batch),
// work-items per group (block), the required sub-group size (sg, 0: IGC's
// choice), the range (dims: 1, or 2 as groups x work-group, or 0 for a plain
// range of one work-item per element, the work-group left to the runtime) and
// floats of local memory per group (lmem, all its local arrays together).
// A kernel declares it as
//   static constexpr bk::kernel_shape shape{batch, block, sg, dims, lmem};
// takes its sub-group size from it in get(properties_tag), and is launched by
// bk::launch, so declaration and launch cannot disagree.
struct kernel_shape {
    int batch;
    int block;
    int sg;
    int dims;
    int lmem;
};

// a shape an Xe-HPG GPU can run: at most 1024 work-items, a multiple of 16
// (whole sub-groups of 16, as block copies need; with sg = 32 the last
// sub-group may be partial), at most 60 KB of local memory
constexpr bool shape_valid(const kernel_shape& s)
{
    if (s.dims == 0) {
        return s.batch == 1 && s.block == 0 && s.lmem == 0;
    }
    return s.batch >= 1 && s.block >= 1 && s.block <= 1024 && s.block % 16 == 0 && (s.dims == 1 || s.dims == 2) &&
           s.lmem >= 0 && s.lmem * int(sizeof(float)) <= 61440;
}

// a local array of n floats, for the kernels' make(cgh) below
inline sycl::local_accessor<float, 1> local_floats(sycl::handler& cgh, const int n)
{
    return sycl::local_accessor<float, 1>(sycl::range<1>(n), cgh);
}

// One launch over nelmt elements in shape s, whose range has Dims
// dimensions: ceil(nelmt / batch) work-groups of s.block work-items (along
// the range's last dimension), or `groups` of them when groups > 0 (kernels
// that stride over the batches).  make(cgh) builds the kernel functor, its
// local arrays included.  For shapes known at run time (specialization
// constants); compile-time shapes go through launch<K> below.
template <int Dims, typename Make>
sycl::event launch(sycl::queue& q, const kernel_shape& s, const int nelmt, Make&& make, const int groups = 0)
{
    assert(s.dims == Dims && shape_valid(s));
    const std::size_t ngroups = (groups > 0) ? std::size_t(groups) : std::size_t((nelmt + s.batch - 1) / s.batch);
    const std::size_t block = s.block;
    return q.submit([&](sycl::handler& cgh) {
        if constexpr (Dims == 0) {
            cgh.parallel_for(sycl::range<1>(std::size_t(nelmt)), make(cgh));
        } else if constexpr (Dims == 2) {
            cgh.parallel_for(sycl::nd_range<2>({ngroups, block}, {1, block}), make(cgh));
        } else {
            cgh.parallel_for(sycl::nd_range<1>(ngroups * block, block), make(cgh));
        }
    });
}

// One launch of kernel K in its shape K::shape, checked at compile time.
template <typename K, typename Make>
sycl::event launch(sycl::queue& q, const int nelmt, Make&& make, const int groups = 0)
{
    static_assert(shape_valid(K::shape), "at most 1024 work-items, a multiple of 16, at most 60 KB of local memory");
    return launch<K::shape.dims>(q, K::shape, nelmt, make, groups);
}

// best host and device times start at max() and drop with each launch;
// device stays max() when profiling is off
struct launch_times {
    double host = std::numeric_limits<double>::max();
    double device = std::numeric_limits<double>::max();
    double first = 0.0;   // host time of the warm-up launch
};

// The GPU queue of a harness.  BK_PROFILE=0: no event profiling (no device
// times, less launch overhead); BK_INORDER=1: an in-order queue.
inline bool profiling() { return get_env("BK_PROFILE").value_or("1") != "0"; }

inline double seconds_since(const std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

inline sycl::queue make_queue()
{
    const bool in_order = get_env("BK_INORDER").value_or("0") == "1";
    if (profiling() && in_order) {
        return sycl::queue{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{},
                                                                     sycl::property::queue::in_order{}}};
    }
    if (profiling()) {
        return sycl::queue{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}}};
    }
    if (in_order) {
        return sycl::queue{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    }
    return sycl::queue{sycl::gpu_selector_v};
}

// launch() submits one run and returns its event.  With profile = false
// (a queue without enable_profiling) only the host time is measured.
template <typename F>
launch_times time_launches(const int ntests, F&& launch, const bool profile = profiling())
{
    launch_times best;
    const auto w0 = std::chrono::steady_clock::now();
    launch().wait_and_throw();   // warm-up (and JIT compilation)
    best.first = seconds_since(w0);
    for (int t = 0; t < ntests; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        sycl::event ev = launch();
        ev.wait_and_throw();
        best.host = std::min(best.host, seconds_since(t0));
        if (profile) {
            const double ns = double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                     ev.get_profiling_info<sycl::info::event_profiling::command_start>());
            best.device = std::min(best.device, 1e-9 * ns);
        }
    }
    return best;
}

// A kernel built for one value of a specialization constant, before any
// timing: the JIT runs here (or the persistent cache answers, see
// scripts/run.sh), and launches pass the bundle to the handler
// (cgh.use_kernel_bundle) instead of setting the constant.  seconds: the
// build time.  Kernel is the kernel's name (its functor type), SpecId its
// specialization_id.  Recorded per kernel in sycl/README.md.
struct specialized_kernel {
    sycl::kernel_bundle<sycl::bundle_state::executable> bundle;
    double seconds;
};

template <typename Kernel, auto& SpecId, typename V>
specialized_kernel build_specialized(const sycl::queue& q, const V& value)
{
    const auto t0 = std::chrono::steady_clock::now();
    auto input = sycl::get_kernel_bundle<sycl::bundle_state::input>(q.get_context(), {q.get_device()},
                                                                     {sycl::get_kernel_id<Kernel>()});
    input.template set_specialization_constant<SpecId>(value);
    auto exe = sycl::build(input);
    return {exe, seconds_since(t0)};
}

// time_launches for one variant of a harness: false, with a "skipped" line,
// when the device cannot run it (a work-group or shared local memory larger
// than the device allows, e.g. 512 work-items on Xe-LP against 1024 on
// Xe-HPG), so the other variants still run.
template <typename F>
bool time_variant(const char* name, const int ntests, F&& launch, launch_times& t, const bool profile = profiling())
{
    try {
        t = time_launches(ntests, launch, profile);
    } catch (const sycl::exception& e) {
        std::printf("%-14s skipped: %s\n", name, e.what());
        return false;
    }
    return true;
}

} // namespace bk

#endif // BK_SYCL_H
