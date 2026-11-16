// bk1_cm.cpp -- host driver of the C for Metal BK1 kernel (cm/bk1_cm_genx.cpp)
// through SYCL (cm_sycl.hpp: the binary is loaded by Level Zero or OpenCL
// interop, the arrays are USM).  The kernel is compiled ahead of time, one
// binary per order, by cm/Makefile.
//
//   build/cm/bk1_cm <p> [nelmt] [ntests]      p = 1..8, nq = p + 2
//
// Loads bk1_cm_<p>.bin, or bk1_cm_<p>_e<n>.bin with BK_EPT=n elements per
// thread (cmsycl::kernel_binary), measures the copy bandwidth (copy_cm), runs
// bk1_cm at each work-group size of $BK_WG (threads per group, default
// "16 32 64"; at p = 8 a pair of threads per element, "4 8 16", even and at
// most 16), and checks every result against the double-precision host
// reference (bk_harness.h).  Data as in the SYCL driver: pseudo-random in and
// JxW, basis cos(i * nq + p).
//
// Each line reports host time (submit to completion) and device time (event
// profiling), the best of ntests runs after a warm-up, in GDoF/s and in GB/s
// of the compulsory traffic (in, out, JxW), and the device GB/s as a share of
// copy_cm's.

#include "cm_sycl.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"

int main(int argc, char** argv)
{
    const int p = (argc > 1) ? std::atoi(argv[1]) : 4;
    const int nq = p + 2;
    const int nm = nq - 1;
    if (p < 1 || p > 8) {
        std::fprintf(stderr, "unsupported p = %d (supported: 1..8)\n", p);
        return 2;
    }
    const std::size_t nelmt = (argc > 2) ? std::size_t(std::atoll(argv[2])) : std::size_t(1) << 19;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 20;
    const int P = (nq <= 4) ? 4 : (nq <= 8) ? 8 : nq;   // as in the kernel
    const int ept = bk::env_int("BK_EPT", 1);           // elements per thread, as compiled
    const std::string bin = cmsycl::kernel_binary("bk1", p, ept);

    // p = 8: a pair of threads per element, work-groups of at most 16
    // threads, even
    const bool pair = (nq == 10);
    const std::vector<int> wgs = cmsycl::work_groups(pair ? "4 8 16" : "16 32 64");
    for (const int wg : wgs) {
        if (pair && (wg % 2 != 0 || wg > 16)) {
            std::fprintf(stderr, "p = 8: the work-group size must be even and at most 16 (BK_WG)\n");
            return 2;
        }
    }

    // setup
    sycl::queue q = cmsycl::make_queue();
    cmsycl::program prog = cmsycl::load_program(q, bin);
    std::printf("BK1 CM: p = %d, nq = %d, nelmt = %zu, ntests = %d, device %s (%s), kernel %s\n", p, nq, nelmt,
                ntests, q.get_device().get_info<sycl::info::device::name>().c_str(), cmsycl::backend_name(q),
                bin.c_str());

    // data and the double-precision reference
    const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
    std::vector<float> JxW(nelmt * nq * nq * nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    bk::fill_random_test_data(in, JxW);
    std::vector<double> ref(in.size());
    bk::reference_bk1(nq, nelmt, basis, JxW, in, ref);
    const double big = bk::max_abs(ref);

    // padded basis: Bf(i, p) = B(i, p) (nm x P), then Bb(p, i) = B(i, p) (nq x P);
    // p = 8 (a pair of threads per element) adds, for t = 0, 1, the copies
    // with thread t's p = {4t..4t+3, 8+t} first: Bf_t (nm x 5, row stride 8),
    // then Bb_t (5 x nm, row stride P), as laid out in the kernel
    const int nb = bk::round_up((nm + nq) * P, 4);
    const int nbft = bk::round_up(nm * 8, 4);
    const int nbbt = bk::round_up(5 * P, 4);
    std::vector<float> bpad(nb + (pair ? 2 * (nbft + nbbt) : 0), 0.0f);
    for (int i = 0; i < nm; ++i) {
        for (int pp = 0; pp < nq; ++pp) {
            bpad[i * P + pp] = basis[i * nq + pp];
            bpad[(nm + pp) * P + i] = basis[i * nq + pp];
        }
    }
    if (pair) {
        for (int t = 0; t < 2; ++t) {
            for (int c = 0; c < 5; ++c) {
                const int pp = (c < 4) ? 4 * t + c : 8 + t;
                for (int i = 0; i < nm; ++i) {
                    bpad[nb + t * nbft + i * 8 + c] = basis[i * nq + pp];
                    bpad[nb + 2 * nbft + t * nbbt + c * P + i] = basis[i * nq + pp];
                }
            }
        }
    }

    // device arrays
    float* d_b = cmsycl::to_device(q, bpad);
    float* d_jxw = cmsycl::to_device(q, JxW);
    float* d_in = cmsycl::to_device(q, in);
    float* d_out = sycl::malloc_device<float>(in.size(), q);
    q.wait();
    const double bytes = sizeof(float) * (2.0 * in.size() + JxW.size());
    const double copy_gbs = cmsycl::copy_bandwidth(q, prog, bytes, ntests);

    // BK1 at each work-group size
    const sycl::kernel bk1 = prog.kernel("bk1_cm");
    const int tpe = pair ? 2 : 1;   // threads per element
    std::vector<float> out(in.size());
    bool ok = true;
    for (const int wg : wgs) {
        q.fill(d_out, 0.0f, out.size()).wait();
        const int global = bk::round_up(int((nelmt + ept - 1) / ept) * tpe, wg);
        const bk::launch_times t = bk::time_launches(ntests, [&] {
            return cmsycl::launch(q, bk1, global, wg, d_b, d_jxw, d_in, d_out, int(nelmt));
        });
        q.memcpy(out.data(), d_out, out.size() * sizeof(float)).wait();
        const double relerr = bk::rel_maxdiff(out, ref, big);
        ok = ok && relerr < 1e-5;
        cmsycl::report("bk1_cm_e" + std::to_string(ept) + "_wg" + std::to_string(wg), in.size(), bytes, t, copy_gbs,
                       relerr);
    }

    for (float* ptr : {d_b, d_jxw, d_in, d_out}) {
        sycl::free(ptr, q);
    }
    return ok ? 0 : 1;
}
