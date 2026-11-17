// bk1_sycl_sc.cpp -- BK1 in SYCL, every order and kernel shape in one binary:
// the opt kernel of sycl/bk1_sycl.cpp (sycl/bk1_sycl_opt.h) with nq, the
// batch, the work-group size and the SLM layout taken from a
// specialization constant, so the JIT compiles each shape with them as
// constants on its first launch.  Built for JIT (spir64) only: ahead of time
// the constant would be a run-time value.
//
//   scripts/build.sh bk1_sycl_sc
//   scripts/run.sh bk1_sycl_sc <p> [nelmt] [ntests]     (nq = p + 2)
//   BK_KERNEL="scP_b1 scF_f128" ...      run a subset (default: all)
//   BK_SHAPE="b4w256p1 b1w128"            run these shapes instead: batch b,
//                                         work-group size w, p1 padded SLM,
//                                         io1 sub-group block copies
//
// Variant names mirror the compile-time ones of bk1_sycl.cpp (inlP_b1 ->
// scP_b1, inlF_f128 -> scF_f128, ...); sc_b<B> is the opt kernel with the
// dense layout at the block of inl_b<B>.  The batch is capped as there
// (bk1gpu::cap_batch).  "jit" is the host time of the first launch, which
// compiles the shape; the rates are the best of ntests launches after it.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "bk_common.h"
#include "bk_harness.h"
#include "bk1_gpu.h"
#include "bk1_sycl_opt.h"

namespace {

using bk1opt::opt_params;

struct shape {
    std::string name;
    opt_params P;
    bool pad;
    bool block_io;
};

constexpr int round16(const int n) { return bk::round_up(n, 16); }

// the shapes of the compile-time opt variants of bk1_sycl.cpp, for nq
std::vector<shape> default_shapes(const int nq)
{
    std::vector<shape> s;
    const auto by_batch = [&](const char* tag, const int want, const bool pad, const bool io) {
        const int b = bk1gpu::cap_batch(nq, want);
        s.push_back({std::string(tag) + "_b" + std::to_string(want),
                     bk1opt::make_opt_params(nq, b, round16(b * nq * nq), pad), pad, io});
    };
    const auto by_fill = [&](const char* tag, const int threads, const bool pad) {
        const int b = bk1gpu::cap_batch(nq, bk1gpu::fill_batch(nq, threads));
        s.push_back({std::string(tag) + "_f" + std::to_string(threads), bk1opt::make_opt_params(nq, b, threads, pad),
                     pad, false});
    };
    for (const int b : {1, 4, 8, 16}) {
        by_batch("sc", b, false, false);
    }
    for (const int b : {1, 4, 8, 16}) {
        by_batch("scP", b, true, false);
    }
    for (const int b : {1, 4, 16}) {
        by_batch("scB", b, false, true);
        by_batch("scBP", b, true, true);
    }
    for (const int t : {128, 256, 512}) {
        by_fill("scF", t, false);
        by_fill("scFP", t, true);
    }
    return s;
}

// BK_SHAPE words "b<batch>w<block>[p<0|1>][io<0|1>]"
std::vector<shape> env_shapes(const int nq, const std::string& list)
{
    std::vector<shape> s;
    std::istringstream ss(list);
    std::string w;
    while (ss >> w) {
        int b = 0;
        int blk = 0;
        int pad = 0;
        int io = 0;
        bool ok = true;
        for (std::size_t i = 0; i < w.size() && ok;) {
            const std::string key = (w.compare(i, 2, "io") == 0) ? "io" : w.substr(i, 1);
            i += key.size();
            std::size_t used = 0;
            int value = 0;
            try {
                value = std::stoi(w.substr(i), &used);
            } catch (const std::exception&) {
                ok = false;
                break;
            }
            i += used;
            if (key == "b") {
                b = value;
            } else if (key == "w") {
                blk = value;
            } else if (key == "p") {
                pad = value;
            } else if (key == "io") {
                io = value;
            } else {
                ok = false;
            }
        }
        if (!ok || b < 1 || blk < 1) {
            std::fprintf(stderr, "BK_SHAPE: cannot parse \"%s\" (b<batch>w<block>[p<0|1>][io<0|1>])\n", w.c_str());
            std::exit(2);
        }
        s.push_back({w, bk1opt::make_opt_params(nq, b, blk, pad != 0), pad != 0, io != 0});
    }
    return s;
}

// two entries for the same kernel (a batch capped to an earlier one's)
bool same_shape(const shape& a, const shape& b)
{
    return a.P.batch == b.P.batch && a.P.block == b.P.block && a.pad == b.pad && a.block_io == b.block_io;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: bk1_sycl_sc <p> [nelmt] [ntests]\n");
        return 2;
    }
    const int nq = std::atoi(argv[1]) + 2;
    const int nm = nq - 1;
    if (nq < 3 || nq > MAXNQ) {
        std::fprintf(stderr, "p = %d: nq = %d outside 3..%d (MAXNQ)\n", nq - 2, nq, MAXNQ);
        return 2;
    }
    const std::size_t nelmt = (argc > 2) ? std::size_t(std::atoll(argv[2])) : std::size_t(1) << 17;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 20;

    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::enable_profiling{}}};
    std::printf("BK1 SYCL sc: p = %d, nq = %d, nelmt = %zu, ntests = %d, device %s\n", nq - 2, nq, nelmt, ntests,
                q.get_device().get_info<sycl::info::device::name>().c_str());

    const std::vector<float> basis = bk::make_test_basis<float>(nm, nq);
    std::vector<float> JxW(nelmt * nq * nq * nq);
    std::vector<float> in(nelmt * nm * nm * nm);
    bk::fill_random_test_data(in, JxW);
    const std::vector<float> JxW_t = bk::transpose_jxw(JxW, nelmt, nq);
    std::vector<double> ref(in.size());
    bk::reference_bk1(nq, nelmt, basis, JxW, in, ref);
    const double big = bk::max_abs(ref);

    float* d_basis = sycl::malloc_device<float>(basis.size(), q);
    float* d_JxW_t = sycl::malloc_device<float>(JxW_t.size(), q);
    float* d_in = sycl::malloc_device<float>(in.size(), q);
    float* d_out = sycl::malloc_device<float>(in.size(), q);
    q.copy(basis.data(), d_basis, basis.size());
    q.copy(JxW_t.data(), d_JxW_t, JxW_t.size());
    q.copy(in.data(), d_in, in.size());
    q.wait_and_throw();

    const std::string shape_env = bk::get_env("BK_SHAPE").value_or("");
    const std::vector<shape> shapes = shape_env.empty() ? default_shapes(nq) : env_shapes(nq, shape_env);
    const std::size_t lmem_max = q.get_device().get_info<sycl::info::device::local_mem_size>();
    const double bytes = sizeof(float) * (2.0 * in.size() + JxW.size());
    std::vector<float> out(in.size());
    for (std::size_t v = 0; v < shapes.size(); ++v) {
        const shape& s = shapes[v];
        if (!bk::selected(s.name.c_str())) {
            continue;
        }
        bool dup = false;
        for (std::size_t u = 0; u < v && !dup; ++u) {
            if (same_shape(shapes[u], s) && bk::selected(shapes[u].name.c_str())) {
                std::printf("%-14s = %s\n", s.name.c_str(), shapes[u].name.c_str());
                dup = true;
            }
        }
        if (dup) {
            continue;
        }
        if (!bk1opt::opt_valid(s.P) || (s.block_io && s.P.block % 16 != 0) ||
            sizeof(float) * bk1opt::opt_lmem(s.P) > lmem_max) {
            std::printf("%-14s skipped: batch %d, block %d does not fit\n", s.name.c_str(), s.P.batch, s.P.block);
            continue;
        }
        const auto launch = [&]() {
            if (s.block_io) {
                return bk1opt::launch_sc<float, true>(q, s.P, d_basis, d_JxW_t, d_in, d_out, int(nelmt));
            }
            return bk1opt::launch_sc<float, false>(q, s.P, d_basis, d_JxW_t, d_in, d_out, int(nelmt));
        };
        q.memset(d_out, 0, out.size() * sizeof(float)).wait();
        double t_first = 0;
        double t_dev = 1e30;
        double t_host = 1e30;
        try {
            const auto t0 = std::chrono::steady_clock::now();
            launch().wait_and_throw();
            t_first = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            for (int t = 0; t < ntests; ++t) {
                const auto t1 = std::chrono::steady_clock::now();
                sycl::event ev = launch();
                ev.wait_and_throw();
                const auto t2 = std::chrono::steady_clock::now();
                t_host = std::min(t_host, std::chrono::duration<double>(t2 - t1).count());
                const double ns = double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                         ev.get_profiling_info<sycl::info::event_profiling::command_start>());
                t_dev = std::min(t_dev, 1e-9 * ns);
            }
        } catch (const sycl::exception& e) {
            std::printf("%-14s skipped: %s\n", s.name.c_str(), e.what());
            continue;
        }
        q.copy(d_out, out.data(), out.size()).wait();
        const double relerr = bk::rel_maxdiff(out, ref, big);
        std::printf("%-14s GDoF/s = %7.3f  GB/s = %6.1f  (device: GDoF/s = %7.3f  GB/s = %6.1f)  relerr = %.1e%s"
                    "  [b%d w%d, jit %.2f s]\n",
                    s.name.c_str(), 1e-9 * in.size() / t_host, 1e-9 * bytes / t_host, 1e-9 * in.size() / t_dev,
                    1e-9 * bytes / t_dev, relerr, bk::fail_mark(relerr), s.P.batch, s.P.block, t_first);
    }
    sycl::free(d_basis, q);
    sycl::free(d_JxW_t, q);
    sycl::free(d_in, q);
    sycl::free(d_out, q);
    return 0;
}
