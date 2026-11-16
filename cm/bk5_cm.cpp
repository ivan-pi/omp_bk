// bk5_cm.cpp -- host driver of the C for Metal BK5 kernel (cm/bk5_cm_genx.cpp,
// symmetric geometric factors G6) through SYCL (cm_sycl.hpp: the binary is
// loaded by Level Zero or OpenCL interop, the arrays are USM).  The kernel is
// compiled ahead of time, one binary per order, by cm/build.sh.
//
//   build/cm/bk5_cm <p> [nelmt] [ntests]      p = 1..8, nq = p + 1
//
// Loads bk5_cm_<p>.bin, or bk5_cm_<p>_e<n>.bin with BK_EPT=n elements per
// thread (cmsycl::kernel_binary), measures the copy bandwidth (copy_cm), runs
// bk5_cm at each work-group size of $BK_WG (threads per group, default
// "16 32 64"; at p = 8 one element per group of nq = 9 threads, "9"), and
// checks every result against the double-precision host reference
// (bk_harness.h).  Data as in the SYCL driver: pseudo-random in and G,
// D(i, n) = cos(i * nq + n), G interleaved by plane rows (layout 1 of
// bk::make_bk5_G: G(e, i, j, f, k)).
//
// Each line reports host time (submit to completion) and device time (event
// profiling), the best of ntests runs after a warm-up, in GDoF/s and in GB/s
// of the compulsory traffic (in, out, six factors per point), and the device
// GB/s as a share of copy_cm's.

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
    const int nq = p + 1;
    if (p < 1 || p > 8) {
        std::fprintf(stderr, "unsupported p = %d (supported: 1..8)\n", p);
        return 2;
    }
    const int nq3 = nq * nq * nq;
    const std::size_t nelmt = (argc > 2) ? std::size_t(std::atoll(argv[2])) : std::size_t(1) << 17;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 20;
    const int P = (nq <= 4) ? 4 : (nq <= 8) ? 8 : 16;   // as in the kernel
    const int ept = bk::env_int("BK_EPT", 1);           // elements per thread, as compiled
    const std::string bin = cmsycl::kernel_binary("bk5", p, ept);

    // p = 8: one element per work-group of nq threads
    const bool slm = (nq == 9);
    const std::vector<int> wgs = cmsycl::work_groups(slm ? "9" : "16 32 64");
    if (slm && ept != 1) {
        std::fprintf(stderr, "p = 8: one element per work-group (EPT 1)\n");
        return 2;
    }
    for (const int wg : wgs) {
        if (slm && wg != nq) {
            std::fprintf(stderr, "p = 8: the work-group size must be %d (BK_WG)\n", nq);
            return 2;
        }
    }

    // setup
    sycl::queue q = cmsycl::make_queue();
    cmsycl::program prog = cmsycl::load_program(q, bin);
    std::printf("BK5 CM: p = %d, nq = %d, nelmt = %zu (%zu DoFs), ntests = %d, device %s (%s), kernel %s\n", p, nq,
                nelmt, nelmt * nq3, ntests, q.get_device().get_info<sycl::info::device::name>().c_str(),
                cmsycl::backend_name(q), bin.c_str());

    // data and the double-precision reference
    const std::vector<float> D = bk::make_test_basis<float>(nq, nq);
    std::vector<float> in(nelmt * nq3);
    std::vector<float> G6i(nelmt * 6 * nq3);
    bk::fill_random_test_data(in, G6i);   // in in [-1, 1], G6i = 1.5 + [-1, 1]
    std::vector<double> ref(in.size());
    bk::reference_bk5(nq, nelmt, D, G6i, in, ref);
    const double big = bk::max_abs(ref);
    const std::vector<float> G = bk::make_bk5_G(G6i, nelmt, nq, 6, 1);
    G6i = std::vector<float>();

    // padded basis: Drow(n, k) = D(n, k), then Dcol(n, k) = D(k, n), rows of P
    std::vector<float> dpad(bk::round_up(2 * nq * P, 4), 0.0f);
    for (int n = 0; n < nq; ++n) {
        for (int k = 0; k < nq; ++k) {
            dpad[n * P + k] = D[n * nq + k];
            dpad[(nq + n) * P + k] = D[k * nq + n];
        }
    }

    // device arrays
    float* d_d = cmsycl::to_device(q, dpad);
    float* d_g = cmsycl::to_device(q, G);
    float* d_in = cmsycl::to_device(q, in);
    float* d_out = sycl::malloc_device<float>(in.size(), q);
    q.wait();
    const double bytes = sizeof(float) * (2.0 * in.size() + G.size());
    const double copy_gbs = cmsycl::copy_bandwidth(q, prog, bytes, ntests);

    // BK5 at each work-group size
    const sycl::kernel bk5 = prog.kernel("bk5_cm");
    std::vector<float> out(in.size());
    bool ok = true;
    for (const int wg : wgs) {
        q.fill(d_out, 0.0f, out.size()).wait();
        const int nthreads = slm ? int(nelmt) * nq : int((nelmt + ept - 1) / ept);
        const int global = bk::round_up(nthreads, wg);
        const bk::launch_times t = bk::time_launches(ntests, [&] {
            return cmsycl::launch(q, bk5, global, wg, d_d, d_g, d_in, d_out, int(nelmt));
        });
        q.memcpy(out.data(), d_out, out.size() * sizeof(float)).wait();
        const double relerr = bk::rel_maxdiff(out, ref, big);
        ok = ok && relerr < 1e-5;
        cmsycl::report("bk5_cm_e" + std::to_string(ept) + "_wg" + std::to_string(wg), in.size(), bytes, t, copy_gbs,
                       relerr);
    }

    for (float* ptr : {d_d, d_g, d_in, d_out}) {
        sycl::free(ptr, q);
    }
    return ok ? 0 : 1;
}
