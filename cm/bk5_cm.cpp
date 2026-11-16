// bk5_cm.cpp -- host driver of the C for Metal BK5 kernel (cm/bk5_cm_genx.cpp,
// symmetric geometric factors G6) through SYCL (cm_sycl.hpp: the binary is
// loaded by Level Zero or OpenCL interop, the arrays are USM).  The kernel is
// compiled ahead of time, one binary per order, by cm/build.sh.
//
//   build/cm/bk5_cm <p> [nelmt] [ntests]      p = 1..8, nq = p + 1
//
// Loads <dir>/bk5_cm_<p>.bin, or bk5_cm_<p>_e<n>.bin with BK_EPT=n elements
// per thread (dir: $BK_CM_DIR, default the executable's directory), runs the
// memory yardstick copy_cm and bk5_cm at each work-group size of $BK_WG
// (threads per group, default "16 32 64"; at p = 8 one element per group of
// nq = 9 threads, "9"), and checks every result against the double-precision host reference
// (bk_harness.h).  Data as in the SYCL driver: pseudo-random in and G,
// D(i, n) = cos(i * nq + n), G interleaved by plane rows (layout 1 of
// bk::make_bk5_G: G(e, i, j, f, k)).
//
// The kernel reads past the last element (the prefetch of the next one, whole
// OWords); unlike OpenCL buffers USM returns no zeros there, so in and G are
// padded with zero elements.
//
// Each line reports host time (submit to completion) and device time (event
// profiling), the best of ntests runs after a warm-up, in GDoF/s and in GB/s
// of the compulsory traffic (in, out, six factors per point), and the device
// GB/s as a share of copy_cm's.

#include "cm_sycl.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
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
    const std::string dir = bk::get_env("BK_CM_DIR").value_or(cmsycl::exe_dir());
    const int ept = bk::env_int("BK_EPT", 1);   // elements per thread, as compiled
    const std::string bin = dir + "/bk5_cm_" + std::to_string(p) + (ept == 1 ? "" : "_e" + std::to_string(ept)) + ".bin";
    // p = 8: a work-group of nq threads per element
    const bool slm = (nq == 9);
    std::vector<int> wgs;
    {
        std::istringstream ws(bk::get_env("BK_WG").value_or(slm ? "9" : "16 32 64"));
        for (int wg = 0; ws >> wg;) {
            if (slm && (wg != nq || ept != 1)) {
                std::fprintf(stderr, "p = 8: the work-group size must be %d (BK_WG) and EPT 1\n", nq);
                return 2;
            }
            wgs.push_back(wg);
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
    const int nd = (2 * nq * P + 3) / 4 * 4;
    std::vector<float> dpad(nd, 0.0f);
    for (int n = 0; n < nq; ++n) {
        for (int k = 0; k < nq; ++k) {
            dpad[n * P + k] = D[n * nq + k];
            dpad[(nq + n) * P + k] = D[k * nq + n];
        }
    }

    // device arrays; zero elements past the end cover what the kernel reads
    // beyond nelmt: up to EPT work-groups of elements, plus one for the
    // OWord-rounded reads of the last element
    const int wg_max = wgs.empty() ? 1 : *std::max_element(wgs.begin(), wgs.end());
    const std::size_t pad = std::size_t(ept * wg_max + 1);
    float* d_d = cmsycl::to_device(q, dpad);
    float* d_g = cmsycl::to_device(q, G, pad * 6 * nq3);
    float* d_in = cmsycl::to_device(q, in, pad * nq3);
    float* d_out = sycl::malloc_device<float>(in.size() + pad * nq3, q);
    const double bytes = sizeof(float) * (2.0 * in.size() + G.size());

    // memory yardstick: copy the same number of bytes as BK5 moves
    constexpr int copy_chunk = 256;   // COPY_CHUNK of cm_common.h
    constexpr int copy_wg = 64;
    const std::size_t copy_n = std::size_t(bytes / sizeof(float) / 2) / (copy_chunk * copy_wg) * (copy_chunk * copy_wg);
    float* d_src = sycl::malloc_device<float>(copy_n, q);
    float* d_dst = sycl::malloc_device<float>(copy_n, q);
    q.fill(d_src, 1.0f, copy_n).wait();
    const sycl::kernel copy_cm = prog.kernel("copy_cm");
    const cmsycl::times tc = cmsycl::time_launches(ntests, [&] {
        return cmsycl::launch(q, copy_cm, copy_n / copy_chunk, copy_wg, d_src, d_dst);
    });
    const double copy_gbs = 1e-9 * 2.0 * copy_n * sizeof(float) / tc.device;
    std::printf("%-14s GB/s = %6.1f  (device: GB/s = %6.1f)\n", "copy_cm", 1e-9 * 2.0 * copy_n * sizeof(float) / tc.host,
                copy_gbs);

    // BK5 at each work-group size
    const sycl::kernel bk5 = prog.kernel("bk5_cm");
    std::vector<float> out(in.size());
    bool ok = true;
    for (const int wg : wgs) {
        q.fill(d_out, 0.0f, out.size()).wait();
        const std::size_t nthreads = slm ? nelmt * nq : (nelmt + ept - 1) / ept;
        const std::size_t global = cmsycl::round_up_to(nthreads, wg);
        const cmsycl::times t = cmsycl::time_launches(ntests, [&] {
            return cmsycl::launch(q, bk5, global, wg, d_d, d_g, d_in, d_out, int(nelmt));
        });
        q.memcpy(out.data(), d_out, out.size() * sizeof(float)).wait();
        const double relerr = bk::rel_maxdiff(out, ref, big);
        ok = ok && relerr < 1e-5;
        const std::string name = "bk5_cm_e" + std::to_string(ept) + "_wg" + std::to_string(wg);
        std::printf("%-14s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f = %3.0f%% of copy_cm)"
                    "  relerr = %.1e%s\n",
                    name.c_str(), 1e-9 * in.size() / t.host, 1e-9 * bytes / t.host, 1e-9 * in.size() / t.device,
                    1e-9 * bytes / t.device, 100.0 * (bytes / t.device) * 1e-9 / copy_gbs,
                    relerr, bk::fail_mark(relerr));
    }

    for (float* ptr : {d_d, d_g, d_in, d_out, d_src, d_dst}) {
        sycl::free(ptr, q);
    }
    return ok ? 0 : 1;
}
