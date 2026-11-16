// bk1_cm.cpp -- host driver of the C for Metal BK1 kernel (cm/bk1_cm_genx.cpp)
// through OpenCL (the CLHPP C++ bindings).  The kernel is compiled ahead of
// time, one binary per order, by cm/build.sh.
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
// Each line reports host time (enqueue to completion) and device time (event
// profiling), the best of ntests runs after a warm-up, in GDoF/s and in GB/s
// of the compulsory traffic (in, out, JxW), and the device GB/s as a share of
// copy_cm's.

#include "cm_ocl.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "bk_common.h"
#include "bk_harness.h"

namespace {

struct times {
    double host;
    double device;
};

// one warm-up launch, then the best host and device time of ntests launches
template <typename F>
times time_launches(const int ntests, F&& launch)
{
    launch().wait();
    times best{std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    for (int t = 0; t < ntests; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        cl::Event ev = launch();
        ev.wait();
        const auto t1 = std::chrono::steady_clock::now();
        best.host = std::min(best.host, std::chrono::duration<double>(t1 - t0).count());
        const double ns = double(ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                                 ev.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        best.device = std::min(best.device, 1e-9 * ns);
    }
    return best;
}

std::string exe_dir()
{
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return ".";
    }
    std::string s(buf, std::size_t(n));
    return s.substr(0, s.rfind('/'));
}

int round_up_to(const std::size_t n, const int m) { return int((n + m - 1) / m * m); }

} // namespace

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
    const std::string dir = bk::get_env("BK_CM_DIR").value_or(exe_dir());
    const int ept = bk::env_int("BK_EPT", 1);   // elements per thread, as compiled
    const std::string bin = dir + "/bk1_cm_" + std::to_string(p) + (ept == 1 ? "" : "_e" + std::to_string(ept)) + ".bin";

    // setup
    cl::Device dev = cmocl::pick_gpu();
    cl::Context ctx(dev);
    cl::CommandQueue q(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
    cl::Program prog = cmocl::load_program(ctx, dev, bin);
    std::printf("BK1 CM: p = %d, nq = %d, nelmt = %zu, ntests = %d, device %s, kernel %s\n", p, nq, nelmt,
                ntests, dev.getInfo<CL_DEVICE_NAME>().c_str(), bin.c_str());

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
    const bool pair = (nq == 10) && bk::env_int("BK_PAIR", 1) != 0;   // as compiled (PAIR_P8)
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

    cl::Buffer d_b(ctx, bpad.begin(), bpad.end(), true);
    cl::Buffer d_jxw(ctx, JxW.begin(), JxW.end(), true);
    cl::Buffer d_in(ctx, in.begin(), in.end(), true);
    cl::Buffer d_out(ctx, CL_MEM_READ_WRITE, in.size() * sizeof(float));
    const double bytes = sizeof(float) * (2.0 * in.size() + JxW.size());

    // memory yardstick: copy the same number of bytes as BK1 moves
    constexpr int copy_chunk = 256;   // COPY_CHUNK of the kernel
    constexpr int copy_wg = 64;
    const std::size_t copy_n = std::size_t(bytes / sizeof(float) / 2) / (copy_chunk * copy_wg) * (copy_chunk * copy_wg);
    cl::Buffer d_src(ctx, CL_MEM_READ_WRITE, copy_n * sizeof(float));
    cl::Buffer d_dst(ctx, CL_MEM_READ_WRITE, copy_n * sizeof(float));
    q.enqueueFillBuffer(d_src, 1.0f, 0, copy_n * sizeof(float));
    q.finish();
    cl::KernelFunctor<cl::Buffer, cl::Buffer> copy_cm(prog, "copy_cm");
    const times tc = time_launches(ntests, [&] {
        return copy_cm(cl::EnqueueArgs(q, cl::NDRange(copy_n / copy_chunk), cl::NDRange(copy_wg)), d_src, d_dst);
    });
    const double copy_gbs = 1e-9 * 2.0 * copy_n * sizeof(float) / tc.device;
    std::printf("%-14s GB/s = %6.1f  (device: GB/s = %6.1f)\n", "copy_cm", 1e-9 * 2.0 * copy_n * sizeof(float) / tc.host,
                copy_gbs);

    // BK1 at each work-group size
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> bk1(prog, "bk1_cm");
    // threads per element: 2 at p = 8 (work-groups of at most 16 threads, even)
    const int tpe = pair ? 2 : 1;
    std::istringstream wgs(bk::get_env("BK_WG").value_or(pair ? "4 8 16" : "16 32 64"));
    std::vector<float> out(in.size());
    bool ok = true;
    for (int wg = 0; wgs >> wg;) {
        if (pair && (wg % 2 != 0 || wg > 16)) {
            std::fprintf(stderr, "p = 8: the work-group size must be even and at most 16 (BK_WG)\n");
            return 2;
        }
        q.enqueueFillBuffer(d_out, 0.0f, 0, out.size() * sizeof(float));
        q.finish();
        const cl::NDRange global(round_up_to((nelmt + ept - 1) / ept * tpe, wg));
        const times t = time_launches(ntests, [&] {
            return bk1(cl::EnqueueArgs(q, global, cl::NDRange(wg)), d_b, d_jxw, d_in, d_out, int(nelmt));
        });
        cl::copy(q, d_out, out.begin(), out.end());
        const double relerr = bk::rel_maxdiff(out, ref, big);
        ok = ok && relerr < 1e-5;
        const std::string name = "bk1_cm_e" + std::to_string(ept) + "_wg" + std::to_string(wg);
        std::printf("%-14s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f = %3.0f%% of copy_cm)"
                    "  relerr = %.1e%s\n",
                    name.c_str(), 1e-9 * in.size() / t.host, 1e-9 * bytes / t.host, 1e-9 * in.size() / t.device,
                    1e-9 * bytes / t.device, 100.0 * (bytes / t.device) * 1e-9 / copy_gbs, relerr,
                    bk::fail_mark(relerr));
    }
    return ok ? 0 : 1;
}
