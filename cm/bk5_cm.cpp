// bk5_cm.cpp -- host driver of the C for Metal BK5 kernel (cm/bk5_cm_genx.cpp,
// symmetric geometric factors G6) through OpenCL (the CLHPP C++ bindings).
// The kernel is compiled ahead of time, one binary per order, by cm/build.sh.
//
//   build/cm/bk5_cm <p> [nelmt] [ntests]      p = 1..8, nq = p + 1
//
// Loads <dir>/bk5_cm_<p>.bin, or bk5_cm_<p>_e<n>.bin with BK_EPT=n elements
// per thread (dir: $BK_CM_DIR, default the executable's directory), runs the memory yardstick copy_cm and bk5_cm at each
// work-group size of $BK_WG (threads per group, default "16 32 64"), and
// checks every result against the double-precision host reference
// (bk_harness.h).  Data as in the SYCL driver: pseudo-random in and G,
// D(i, n) = cos(i * nq + n), G interleaved by plane rows (layout 1 of
// bk::make_bk5_G: G(e, i, j, f, k)).
//
// Each line reports host time (enqueue to completion) and device time (event
// profiling), the best of ntests runs after a warm-up, in GDoF/s and in GB/s
// of the compulsory traffic (in, out, six factors per point), and the device
// GB/s as a share of copy_cm's.

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
    const int nq = p + 1;
    if (p < 1 || p > 8) {
        std::fprintf(stderr, "unsupported p = %d (supported: 1..8)\n", p);
        return 2;
    }
    const int nq3 = nq * nq * nq;
    const std::size_t nelmt = (argc > 2) ? std::size_t(std::atoll(argv[2])) : std::size_t(1) << 17;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 20;
    const int P = (nq <= 4) ? 4 : (nq <= 8) ? 8 : 16;   // as in the kernel
    const std::string dir = bk::get_env("BK_CM_DIR").value_or(exe_dir());
    const int ept = bk::env_int("BK_EPT", 1);   // elements per thread, as compiled
    const std::string bin = dir + "/bk5_cm_" + std::to_string(p) + (ept == 1 ? "" : "_e" + std::to_string(ept)) + ".bin";

    // setup
    cl::Device dev = cmocl::pick_gpu();
    cl::Context ctx(dev);
    cl::CommandQueue q(ctx, dev, CL_QUEUE_PROFILING_ENABLE);
    cl::Program prog = cmocl::load_program(ctx, dev, bin);
    std::printf("BK5 CM: p = %d, nq = %d, nelmt = %zu (%zu DoFs), ntests = %d, device %s, kernel %s\n", p, nq,
                nelmt, nelmt * nq3, ntests, dev.getInfo<CL_DEVICE_NAME>().c_str(), bin.c_str());

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

    cl::Buffer d_d(ctx, dpad.begin(), dpad.end(), true);
    cl::Buffer d_g(ctx, G.begin(), G.end(), true);
    cl::Buffer d_in(ctx, in.begin(), in.end(), true);
    cl::Buffer d_out(ctx, CL_MEM_READ_WRITE, in.size() * sizeof(float));
    const double bytes = sizeof(float) * (2.0 * in.size() + G.size());

    // memory yardstick: copy the same number of bytes as BK5 moves
    constexpr int copy_chunk = 256;   // COPY_CHUNK of cm_common.h
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

    // BK5 at each work-group size
    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> bk5(prog, "bk5_cm");
    std::istringstream wgs(bk::get_env("BK_WG").value_or("16 32 64"));
    std::vector<float> out(in.size());
    bool ok = true;
    for (int wg = 0; wgs >> wg;) {
        q.enqueueFillBuffer(d_out, 0.0f, 0, out.size() * sizeof(float));
        q.finish();
        const cl::NDRange global(round_up_to((nelmt + ept - 1) / ept, wg));
        const times t = time_launches(ntests, [&] {
            return bk5(cl::EnqueueArgs(q, global, cl::NDRange(wg)), d_d, d_g, d_in, d_out, int(nelmt));
        });
        cl::copy(q, d_out, out.begin(), out.end());
        const double relerr = bk::rel_maxdiff(out, ref, big);
        ok = ok && relerr < 1e-5;
        const std::string name = "bk5_cm_e" + std::to_string(ept) + "_wg" + std::to_string(wg);
        std::printf("%-14s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f = %3.0f%% of copy_cm)"
                    "  relerr = %.1e%s\n",
                    name.c_str(), 1e-9 * in.size() / t.host, 1e-9 * bytes / t.host, 1e-9 * in.size() / t.device,
                    1e-9 * bytes / t.device, 100.0 * (bytes / t.device) * 1e-9 / copy_gbs,
                    relerr, bk::fail_mark(relerr));
    }
    return ok ? 0 : 1;
}
