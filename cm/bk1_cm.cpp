// bk1_cm.cpp -- host driver of the C for Metal BK1 kernel (cm/bk1_cm_genx.cpp)
// through SYCL (cm_sycl.hpp: the binary is loaded by Level Zero or OpenCL
// interop, the arrays are USM).  The kernel is compiled ahead of time, one
// binary per order, by cm/build.sh.
//
//   build/cm/bk1_cm <p> [nelmt] [ntests]      p = 1..8, nq = p + 2
//
// Loads <dir>/bk1_cm_<p>.bin, or bk1_cm_<p>_e<n>.bin with BK_EPT=n elements
// per thread (dir: $BK_CM_DIR, default the executable's directory), runs the
// memory yardstick copy_cm and bk1_cm at each work-group size of $BK_WG
// (threads per group, default "16 32 64"; at p = 8 a pair of threads per
// element, "4 8 16", even and at most 16; BK_PAIR=0 for a kernel built with
// PAIR_P8=0, one thread per element), and checks every result against the
// double-precision host reference (bk_harness.h).  Data as in the SYCL
// driver: pseudo-random in and JxW, basis cos(i * nq + p).
//
// The kernel reads past the last element (the prefetch of the next one with
// EPT > 1, whole OWords); unlike OpenCL buffers USM returns no zeros there,
// so in and JxW are padded with zero elements.
//
// Each line reports host time (submit to completion) and device time (event
// profiling), the best of ntests runs after a warm-up, in GDoF/s and in GB/s
// of the compulsory traffic (in, out, JxW), and the device GB/s as a share of
// copy_cm's.

#include "cm_sycl.hpp"

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
    const int nq = p + 2;
    const int nm = nq - 1;
    if (p < 1 || p > 8) {
        std::fprintf(stderr, "unsupported p = %d (supported: 1..8)\n", p);
        return 2;
    }
    const std::size_t nelmt = (argc > 2) ? std::size_t(std::atoll(argv[2])) : std::size_t(1) << 19;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 20;
    const int P = (nq <= 4) ? 4 : (nq <= 8) ? 8 : nq;   // as in the kernel
    const std::string dir = bk::get_env("BK_CM_DIR").value_or(cmsycl::exe_dir());
    const int ept = bk::env_int("BK_EPT", 1);   // elements per thread, as compiled
    const std::string bin = dir + "/bk1_cm_" + std::to_string(p) + (ept == 1 ? "" : "_e" + std::to_string(ept)) + ".bin";

    // p = 8: a pair of threads per element (as compiled, PAIR_P8), work-groups
    // of at most 16 threads, even
    const bool pair = (nq == 10) && bk::env_int("BK_PAIR", 1) != 0;
    std::vector<int> wgs;
    {
        std::istringstream ws(bk::get_env("BK_WG").value_or(pair ? "4 8 16" : "16 32 64"));
        for (int wg = 0; ws >> wg;) {
            if (pair && (wg % 2 != 0 || wg > 16)) {
                std::fprintf(stderr, "p = 8: the work-group size must be even and at most 16 (BK_WG)\n");
                return 2;
            }
            wgs.push_back(wg);
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
    // BK_ZERO_P="a b": zero the weights of those p (checks one thread of a
    // p = 8 pair alone: "0 1 2 3 8" leaves thread 1's p, "4 5 6 7 9" thread 0's)
    {
        std::istringstream zs(bk::get_env("BK_ZERO_P").value_or(""));
        for (int zp = 0; zs >> zp;) {
            for (std::size_t e = 0; e < nelmt; ++e) {
                for (int x = 0; x < nq * nq; ++x) {
                    JxW[(e * nq + zp) * nq * nq + x] = 0.0f;
                }
            }
        }
    }
    std::vector<double> ref(in.size());
    bk::reference_bk1(nq, nelmt, basis, JxW, in, ref);
    const double big = bk::max_abs(ref);

    // padded basis: Bf(i, p) = B(i, p) (nm x P), then Bb(p, i) = B(i, p) (nq x P);
    // p = 8 (a pair of threads per element) adds, for t = 0, 1, the copies
    // with thread t's p = {4t..4t+3, 8+t} first: Bf_t (nm x 5, row stride 8),
    // then Bb_t (5 x nm, row stride P), as laid out in the kernel
    const auto r4 = [](const int n) { return (n + 3) / 4 * 4; };
    const int off_t = r4((nm + nq) * P);
    const int nbft = r4(nm * 8);
    const int nbbt = r4(5 * P);
    std::vector<float> bpad(pair ? off_t + 2 * (nbft + nbbt) : r4((nm + nq) * P), 0.0f);
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
                    bpad[off_t + t * nbft + i * 8 + c] = basis[i * nq + pp];
                    bpad[off_t + 2 * nbft + t * nbbt + c * P + i] = basis[i * nq + pp];
                }
            }
        }
    }

    // device arrays; zero elements past the end cover what the kernel reads
    // beyond nelmt: the next element of the last thread, plus one for the
    // OWord-rounded reads of the last element
    const std::size_t pad = std::size_t(ept + 1);
    float* d_b = cmsycl::to_device(q, bpad);
    float* d_jxw = cmsycl::to_device(q, JxW, pad * nq * nq * nq);
    float* d_in = cmsycl::to_device(q, in, pad * nm * nm * nm);
    float* d_out = sycl::malloc_device<float>(in.size() + pad * nm * nm * nm, q);
    const double bytes = sizeof(float) * (2.0 * in.size() + JxW.size());

    // memory yardstick: copy the same number of bytes as BK1 moves
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

    // BK1 at each work-group size
    const sycl::kernel bk1 = prog.kernel("bk1_cm");
    const int tpe = pair ? 2 : 1;   // threads per element
    std::vector<float> out(in.size());
    bool ok = true;
    for (const int wg : wgs) {
        q.fill(d_out, 0.0f, out.size()).wait();
        const std::size_t global = cmsycl::round_up_to((nelmt + ept - 1) / ept * tpe, wg);
        const cmsycl::times t = cmsycl::time_launches(ntests, [&] {
            return cmsycl::launch(q, bk1, global, wg, d_b, d_jxw, d_in, d_out, int(nelmt));
        });
        q.memcpy(out.data(), d_out, out.size() * sizeof(float)).wait();
        const double relerr = bk::rel_maxdiff(out, ref, big);
        ok = ok && relerr < 1e-5;
        const std::string name = "bk1_cm_e" + std::to_string(ept) + "_wg" + std::to_string(wg);
        std::printf("%-14s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f = %3.0f%% of copy_cm)"
                    "  relerr = %.1e%s\n",
                    name.c_str(), 1e-9 * in.size() / t.host, 1e-9 * bytes / t.host, 1e-9 * in.size() / t.device,
                    1e-9 * bytes / t.device, 100.0 * (bytes / t.device) * 1e-9 / copy_gbs, relerr,
                    bk::fail_mark(relerr));
    }

    for (float* ptr : {d_b, d_jxw, d_in, d_out, d_src, d_dst}) {
        sycl::free(ptr, q);
    }
    return ok ? 0 : 1;
}
