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
#include <chrono>
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
