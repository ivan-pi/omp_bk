// BK1_ff_cuda.cu -- driver for the float-float BK1 kernel of bk1_ff_cuda.cuh
// on the GPU, with the CCCL runtime API of CUDA 13.2+ (cuda::launch,
// cuda::buffer, cuda::stream): the kernel is a functor whose call operator
// takes the launch configuration as a template parameter, reads its block
// and thread ranks and the dynamic shared memory from it, and calls the
// kernel body mass_operator_ff_body.
//
// The driver builds the same test data as BK1_ff (constant 3.0/1.0, or
// pseudo-random with BK_RANDOM=1), computes the double reference on the
// host with SumFactorization<double> of bk1_sumfact.h, times ntests runs of
// the kernel and prints GDoF/s, GB/s (8 bytes per value in, out and JxW),
// the norm, and the relative L2 error against the reference, ending with
// ok or FAIL (exit status 1) against BK_TOL (default 1e-10).  The host
// reference is single-threaded and takes a while at the default element
// count and high orders.
// From the repository root:  make BK1_ff_cuda   (needs nvcc with the CCCL
// of CUDA 13.2 or later; CCCL=/path/to/a/cccl/checkout puts a newer CCCL
// ahead of the toolkit's)
//
// Run:  ./BK1_ff_cuda [p=2] [nelmt=524288] [ntests=5] [nelmtPerBatch]
//   nelmtPerBatch  elements per block (default: fills about 128 threads,
//                  mass_operator_ff_traits<nq>::default_batch()); the block
//                  has nelmtPerBatch * nq * nq threads rounded up to warps.
//   BK_RANDOM=1, BK_TOL=t  as in BK1_ff.

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/chrono>
#include <cuda/std/span>
#include <cuda/stream>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "bk1_ff_cuda.cuh"
#include "bk_common.h"
#include "bk1_sumfact.h"

namespace {

// The kernel functor: the configuration is its first argument, from which
// it takes the block and thread ranks and the dynamic shared memory
// (cuda::dynamic_shared_memory<float[]> in the configuration).
template <int nq>
struct MassOperatorFFKernel {
    template <typename Config>
    __device__ void operator()(const Config config,
                               const unsigned int nelmt, const unsigned int nelmtPerBatch,
                               cuda::std::span<const double> basis,
                               cuda::std::span<const double> JxW,
                               cuda::std::span<const double> in,
                               cuda::std::span<double> out) const
    {
        const cuda::std::span<float> shared = cuda::dynamic_shared_memory(config);
        bk::mass_operator_ff_body<nq>(
            nelmt, nelmtPerBatch, basis.data(), JxW.data(), in.data(), out.data(), shared.data(),
            static_cast<unsigned int>(cuda::block.rank(cuda::grid, config)),
            static_cast<unsigned int>(cuda::block.count(cuda::grid, config)),
            static_cast<unsigned int>(cuda::gpu_thread.rank(cuda::block, config)),
            static_cast<unsigned int>(cuda::gpu_thread.count(cuda::block, config)));
    }
};

// ||x - ref|| / ||ref|| in double.
double relative_error(const std::vector<double>& x, const std::vector<double>& ref)
{
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = x[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num) / std::sqrt(den);
}

template <int nq>
bool run_test(const std::size_t nelmt, const int ntests, const unsigned int nelmtPerBatch,
              const bool random_data, const double tol)
{
    using traits = bk::mass_operator_ff_traits<nq>;
    constexpr int nm = traits::nm;
    constexpr int nq3 = traits::nq3;
    constexpr int nm3 = traits::nm3;

    // Test data and the double reference on the host.
    const std::array<double, nm * nq> b = bk::make_test_basis<double, nm, nq>();
    const std::vector<double> basis(b.begin(), b.end());
    std::vector<double> JxW(nelmt * nq3, 1.0);
    std::vector<double> in(nelmt * nm3, 3.0);
    if (random_data) {
        bk::fill_random_test_data(in, JxW);
    }
    std::vector<double> ref(in.size());
    bk::SumFactorization<double, nq>(nelmt, basis.data(), JxW.data(), in.data(), ref.data());

    // Device buffers, copied from the host data in stream order.
    cuda::device_ref device = cuda::devices[0];
    cuda::stream stream{device};
    auto pool = cuda::device_default_memory_pool(device);
    auto d_basis = cuda::make_buffer<double>(stream, pool, basis);
    auto d_JxW   = cuda::make_buffer<double>(stream, pool, JxW);
    auto d_in    = cuda::make_buffer<double>(stream, pool, in);
    auto d_out   = cuda::make_buffer<double>(stream, pool, in.size(), cuda::no_init);
    stream.sync();

    // Launch configuration: a block per batch of elements, every point of
    // the nq x nq plane a thread, a grid-stride loop over the batches.
    const std::size_t nbatches = (nelmt + nelmtPerBatch - 1) / nelmtPerBatch;
    const int sms = cuda::device_attributes::multiprocessor_count(device);
    const int nblocks = static_cast<int>(std::min<std::size_t>(nbatches, static_cast<std::size_t>(sms) * 32));
    const int block = traits::block_size(static_cast<int>(nelmtPerBatch));
    const std::size_t smem_floats = traits::shared_floats(static_cast<int>(nelmtPerBatch));
    const auto config = cuda::make_config(
        cuda::grid_dims(nblocks), cuda::block_dims(block),
        cuda::dynamic_shared_memory<float[]>(smem_floats, cuda::non_portable));

    std::cout << "SumFactorization -> nelmt = " << nelmt << " nq = " << nq
              << " data = " << (random_data ? "random" : "constant")
              << " device = " << device.name()
              << " batch = " << nelmtPerBatch << " block = " << block
              << " grid = " << nblocks << " smem = " << smem_floats * sizeof(float) << " B\n";

    // Minimum time over ntests runs.
    double elapsed = 1e300;
    for (int t = 0; t < ntests; ++t) {
        cuda::timed_event start{stream};
        cuda::launch(stream, config, MassOperatorFFKernel<nq>{},
                     static_cast<unsigned int>(nelmt), nelmtPerBatch, d_basis, d_JxW, d_in, d_out);
        cuda::timed_event stop{stream};
        stop.sync();
        const cuda::std::chrono::nanoseconds ns = stop - start;
        elapsed = std::min(elapsed, 1.0e-9 * static_cast<double>(ns.count()));
    }

    // The result back to the host (cudaMemcpyAsync rather than
    // cuda::copy_bytes, whose header <cuda/algorithm> is newer than some
    // toolkits' CCCL).
    std::vector<double> out(in.size());
    const cudaError_t copy_status = cudaMemcpyAsync(
        out.data(), d_out.data(), out.size() * sizeof(double), cudaMemcpyDeviceToHost, stream.get());
    if (copy_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync: ") + cudaGetErrorString(copy_status));
    }
    stream.sync();

    const double dof_rate  = 1.0e-9 * static_cast<double>(in.size()) / elapsed;
    const double byte_rate = 1.0e-9 * sizeof(double) * static_cast<double>(2 * in.size() + JxW.size()) / elapsed;
    const double err = relative_error(out, ref);
    const bool ok = err <= tol;
    std::cout << std::left << std::setw(12) << "float-float" << std::right
              << " GDoF/s = " << std::setw(9) << dof_rate
              << " GB/s = "   << std::setw(9) << byte_rate
              << " norm = "   << std::setprecision(10) << bk::norm2(out) << std::setprecision(6) << "\n"
              << (ok ? "ok" : "FAIL") << ": float-float rel. error vs double = " << err
              << " (tol = " << tol << ")\n";
    return ok;
}

constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288, as BK1

} // namespace

int main(int argc, char** argv)
try {
    const int p = (argc > 1) ? std::atoi(argv[1]) : 2;
    const std::size_t nelmt =
        (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;
    const unsigned int batch_arg = (argc > 4) ? static_cast<unsigned int>(std::atoi(argv[4])) : 0u;
    const bool random_data = bk::get_env("BK_RANDOM").has_value();
    const double tol = std::atof(bk::get_env("BK_TOL").value_or("1e-10").c_str());

    if (cuda::devices.size() == 0) {
        std::cerr << "no CUDA device\n";
        return 1;
    }

    // Runtime p -> compile-time nq = p + 2, one instantiation per order.
    const auto run = [&](auto nq_tag) {
        constexpr int nq = decltype(nq_tag)::value;
        const unsigned int batch = (batch_arg > 0)
            ? batch_arg : static_cast<unsigned int>(bk::mass_operator_ff_traits<nq>::default_batch());
        return run_test<nq>(nelmt, ntests, batch, random_data, tol);
    };
    bool ok = false;
    switch (p) {
        case 1: ok = run(std::integral_constant<int,  3>{}); break;
        case 2: ok = run(std::integral_constant<int,  4>{}); break;
        case 3: ok = run(std::integral_constant<int,  5>{}); break;
        case 4: ok = run(std::integral_constant<int,  6>{}); break;
        case 5: ok = run(std::integral_constant<int,  7>{}); break;
        case 6: ok = run(std::integral_constant<int,  8>{}); break;
        case 7: ok = run(std::integral_constant<int,  9>{}); break;
        case 8: ok = run(std::integral_constant<int, 10>{}); break;
        default:
            std::cerr << "unsupported polynomial order p = " << p
                      << " (supported: 1..8)\n";
            return 1;
    }
    return ok ? 0 : 1;
}
catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
