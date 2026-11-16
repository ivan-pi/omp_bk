// bk_sycl.h -- timing of the SYCL harnesses (sycl/, tinytc/): one warm-up
// launch, then the best host time (submit to completion) and the best
// device time (event profiling) over ntests launches.

#ifndef BK_SYCL_H
#define BK_SYCL_H

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>

namespace bk {

struct launch_times {
    double host;
    double device;   // max() when profiling is off
};

// launch() submits one run and returns its event.  With profile = false
// (a queue without enable_profiling) only the host time is measured.
template <typename F>
launch_times time_launches(const int ntests, F&& launch, const bool profile = true)
{
    launch().wait_and_throw();   // warm-up (and JIT compilation)
    launch_times best{std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    for (int t = 0; t < ntests; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        sycl::event ev = launch();
        ev.wait_and_throw();
        const auto t1 = std::chrono::steady_clock::now();
        best.host = std::min(best.host, std::chrono::duration<double>(t1 - t0).count());
        if (profile) {
            const double ns = double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                     ev.get_profiling_info<sycl::info::event_profiling::command_start>());
            best.device = std::min(best.device, 1e-9 * ns);
        }
    }
    return best;
}

// time_launches for one variant of a harness: false, with a "skipped" line,
// when the device cannot run it (a work-group or shared local memory larger
// than the device allows, e.g. 512 work-items on Xe-LP against 1024 on
// Xe-HPG), so the other variants still run.
template <typename F>
bool time_variant(const char* name, const int ntests, F&& launch, launch_times& t, const bool profile = true)
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
