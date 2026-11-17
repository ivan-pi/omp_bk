// bk1_sycl_harness.h -- host side of the BK1 SYCL drivers (bk1_sycl.cpp,
// bk1_sycl_sc.cpp): the test data, its double-precision reference, the
// device arrays, and one timed and checked run per selected variant.
//
// Data: pseudo-random in and JxW (bk::fill_random_test_data), basis
// cos(i * nq + p).  The check is against a double-precision CPU reference:
// the largest difference relative to the largest |reference| (float
// rounding gives ~1e-7; an indexing error gives ~1).  JxW is on the device
// transposed per element, JxW_t(e, r, q, p) = JxW(e, p, q, r).

#ifndef BK1_SYCL_HARNESS_H
#define BK1_SYCL_HARNESS_H

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk_sycl.h"
#include "bk1_sycl_opt.h"

namespace bk1sycl {

using device_data = bk1opt::device_data<float>;

struct variant {
    std::string name;
    std::function<sycl::event(sycl::queue&, const device_data&)> fn;
    std::string info;   // printed after the result with the first-launch time, when not empty
};

// Runs the selected variants (BK_KERNEL) on q (bk::make_queue) over nelmt
// elements of order nq and prints one result line each:
//   <name> GDoF/s = ..  GB/s = ..  (device: GDoF/s = ..  GB/s = ..)  relerr = ..
// With BK_CHECK=0 there is no reference, relerr is "off", and JxW goes to the
// device untransposed (random data either way).
inline void run(sycl::queue& q, const char* title, const int nq, const std::size_t nelmt, const int ntests,
                const std::vector<variant>& variants)
{
    const int nm = nq - 1;
    std::printf("%s: p = %d, nq = %d, nelmt = %zu, ntests = %d, device %s\n", title, nq - 2, nq, nelmt, ntests,
                q.get_device().get_info<sycl::info::device::name>().c_str());

    // the host arrays go after the upload: the runs hold only the reference
    // and out (with the check)
    const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    std::vector<float> JxW(nelmt * nq * nq * nq);
    bk::fill_random_test_data(in, JxW);
    const bk::reference_check check(in.size(), [&](std::vector<double>& ref) {
        bk::reference_bk1(nq, nelmt, basis, JxW, in, ref);
    });
    if (check.on()) {
        JxW = bk::transpose_jxw(JxW, nelmt, nq);
    }

    const std::size_t ndofs = in.size();
    const double bytes = sizeof(float) * (2.0 * ndofs + JxW.size());
    const device_data d{sycl::malloc_device<float>(basis.size(), q), sycl::malloc_device<float>(JxW.size(), q),
                        sycl::malloc_device<float>(ndofs, q), sycl::malloc_device<float>(ndofs, q),
                        static_cast<int>(nelmt)};
    q.copy(basis.data(), d.basis, basis.size());
    q.copy(JxW.data(), d.JxW_t, JxW.size());
    q.copy(in.data(), d.in, ndofs);
    q.wait_and_throw();
    std::vector<float>().swap(JxW);
    std::vector<float>().swap(in);

    std::vector<float> out(check.on() ? ndofs : 0);
    for (const variant& v : variants) {
        if (!bk::selected(v.name.c_str())) {
            continue;
        }
        q.memset(d.out, 0, ndofs * sizeof(float)).wait();
        bk::launch_times t;
        if (!bk::time_variant(v.name.c_str(), ntests, [&] { return v.fn(q, d); }, t)) {
            continue;
        }
        const std::string chk = check.text([&]() -> const std::vector<float>& {
            q.copy(d.out, out.data(), ndofs).wait();
            return out;
        });
        std::printf("%-14s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f)  relerr = %s\n",
                    v.name.c_str(), 1e-9 * ndofs / t.host, 1e-9 * bytes / t.host, 1e-9 * ndofs / t.device,
                    1e-9 * bytes / t.device, chk.c_str());
        if (!v.info.empty()) {
            std::printf("%-14s %s first %.3f s\n", v.name.c_str(), v.info.c_str(), t.first);
        }
    }
    sycl::free(d.basis, q);
    sycl::free(d.JxW_t, q);
    sycl::free(d.in, q);
    sycl::free(d.out, q);
}

} // namespace bk1sycl

#endif // BK1_SYCL_HARNESS_H
