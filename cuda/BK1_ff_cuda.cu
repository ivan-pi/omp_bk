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
// From the repository root:  make BK1_ff_cuda   (needs nvcc 12.x or later
// with the CCCL of CUDA 13.2 or later; CCCL=/path/to/a/cccl/checkout puts a
// newer CCCL ahead of the toolkit's).  BK1_ff_cuda_plain.cu is the same
// driver on the classic runtime API, for any nvcc.
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
#include <cuda/std/string_view>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bk1_ff_cuda.cuh"
#include "bk1_ff_driver_common.h"

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

template <int nq>
bool run_test(const bk::ffdrv::Args& args)
{
    using traits = bk::mass_operator_ff_traits<nq>;
    const unsigned int batch = (args.batch > 0) ? args.batch : static_cast<unsigned int>(traits::default_batch());
    const bk::ffdrv::TestData<nq> data(args.nelmt, args.random_data);

    // Device buffers, copied from the host data in stream order.
    cuda::device_ref device = cuda::devices[0];
    cuda::stream stream{device};
    auto pool = cuda::device_default_memory_pool(device);
    auto d_basis = cuda::make_buffer<double>(stream, pool, data.basis);
    auto d_JxW   = cuda::make_buffer<double>(stream, pool, data.JxW);
    auto d_in    = cuda::make_buffer<double>(stream, pool, data.in);
    auto d_out   = cuda::make_buffer<double>(stream, pool, data.in.size(), cuda::no_init);
    stream.sync();

    // Launch configuration: a block per batch of elements, every point of
    // the nq x nq plane a thread, a grid-stride loop over the batches.
    const std::size_t nbatches = (args.nelmt + batch - 1) / batch;
    const int sms = cuda::device_attributes::multiprocessor_count(device);
    const int grid = static_cast<int>(std::min<std::size_t>(nbatches, static_cast<std::size_t>(sms) * 32));
    const int block = traits::block_size(static_cast<int>(batch));
    const std::size_t smem_floats = traits::shared_floats(static_cast<int>(batch));
    const auto config = cuda::make_config(
        cuda::grid_dims(grid), cuda::block_dims(block),
        cuda::dynamic_shared_memory<float[]>(smem_floats, cuda::non_portable));
    const cuda::std::string_view name = device.name();
    bk::ffdrv::print_header(args.nelmt, nq, args.random_data, std::string(name.data(), name.size()),
                            batch, block, grid, smem_floats * sizeof(float));

    // Minimum time over ntests runs.
    double elapsed = 1e300;
    for (int t = 0; t < args.ntests; ++t) {
        cuda::timed_event start{stream};
        cuda::launch(stream, config, MassOperatorFFKernel<nq>{},
                     static_cast<unsigned int>(args.nelmt), batch, d_basis, d_JxW, d_in, d_out);
        cuda::timed_event stop{stream};
        stop.sync();
        const cuda::std::chrono::nanoseconds ns = stop - start;
        elapsed = std::min(elapsed, 1.0e-9 * static_cast<double>(ns.count()));
    }

    // The result back to the host (cudaMemcpyAsync rather than
    // cuda::copy_bytes, whose header <cuda/algorithm> is newer than some
    // toolkits' CCCL).
    std::vector<double> out(data.in.size());
    const cudaError_t copy_status = cudaMemcpyAsync(
        out.data(), d_out.data(), out.size() * sizeof(double), cudaMemcpyDeviceToHost, stream.get());
    if (copy_status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync: ") + cudaGetErrorString(copy_status));
    }
    stream.sync();

    return bk::ffdrv::report("float-float", elapsed, out, data.JxW.size(), data.ref, args.tol);
}

} // namespace

int main(int argc, char** argv)
try {
    const bk::ffdrv::Args args = bk::ffdrv::parse_args(argc, argv);
    if (cuda::devices.size() == 0) {
        std::cerr << "no CUDA device\n";
        return 1;
    }
    const bool ok = bk::ffdrv::dispatch(args.p, [&](auto nq_tag) {
        return run_test<decltype(nq_tag)::value>(args);
    });
    return ok ? 0 : 1;
}
catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
