// BK1_ff_cuda_plain.cu -- driver for the float-float BK1 kernel of
// bk1_ff_cuda.cuh on the GPU with the classic CUDA runtime API only
// (cudaMalloc, cudaMemcpy, <<<grid, block, smem, stream>>>, events), so it
// builds with any nvcc that compiles C++17, CUDA 11.8 included.
// BK1_ff_cuda.cu is the same driver on the CCCL runtime API of CUDA 13.2+.
//
// The driver builds the test data of BK1_ff (constant, or pseudo-random
// with BK_RANDOM=1), computes the double reference on the host, times
// ntests runs of the kernel after warm-up runs of 0.2 s of kernel time
// (BK_WARMUP=s) and prints GDoF/s, GB/s (8 bytes per value in,
// out and JxW), the norm and the relative L2 error against the reference,
// ending with ok or FAIL (exit status 1) against BK_TOL (default 1e-10).
// From the repository root:  make BK1_ff_cuda_plain
//
// Run:  ./BK1_ff_cuda_plain [p=2] [nelmt=524288] [ntests=5] [nelmtPerBatch]
//   nelmtPerBatch  elements per block (default: fills about 128 threads,
//                  mass_operator_ff_traits<nq>::default_batch()); the block
//                  has nelmtPerBatch * nq * nq threads rounded up to warps.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "bk1_ff_cuda.cuh"
#include "bk1_ff_driver_common.h"

namespace {

void check(const cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

// A device array of doubles that frees itself.
struct DeviceArray {
    double* ptr = nullptr;
    std::size_t size = 0;

    explicit DeviceArray(const std::size_t n) : size(n)
    {
        check(cudaMalloc(&ptr, n * sizeof(double)), "cudaMalloc");
    }
    explicit DeviceArray(const std::vector<double>& host) : DeviceArray(host.size())
    {
        check(cudaMemcpy(ptr, host.data(), size * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy to device");
    }
    ~DeviceArray() { cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

template <int nq>
bool run_test(const bk::ffdrv::Args& args)
{
    using traits = bk::mass_operator_ff_traits<nq>;
    const unsigned int batch = (args.batch > 0) ? args.batch : static_cast<unsigned int>(traits::default_batch());
    const bk::ffdrv::TestData<nq> data(args.nelmt, args.random_data);

    DeviceArray d_basis(data.basis);
    DeviceArray d_JxW(data.JxW);
    DeviceArray d_in(data.in);
    DeviceArray d_out(data.in.size());

    // Launch geometry: a block per batch of elements, a thread per point of
    // the nq x nq plane, a grid-stride loop over the batches.
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    const std::size_t nbatches = (args.nelmt + batch - 1) / batch;
    const int grid = static_cast<int>(std::min<std::size_t>(nbatches, static_cast<std::size_t>(prop.multiProcessorCount) * 32));
    const int block = traits::block_size(static_cast<int>(batch));
    const std::size_t smem = traits::shared_bytes(static_cast<int>(batch));
    if (smem > 48 * 1024) {
        check(cudaFuncSetAttribute(bk::MassOperatorFF<nq>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(smem)), "cudaFuncSetAttribute");
    }
    bk::ffdrv::print_header(args.nelmt, nq, args.random_data, prop.name, batch, block, grid, smem);

    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "cudaEventCreate");
    check(cudaEventCreate(&stop), "cudaEventCreate");

    // One run of the kernel, timed with the events, in seconds.
    const auto run_once = [&]() -> double {
        check(cudaEventRecord(start, stream), "cudaEventRecord");
        bk::MassOperatorFF<nq><<<grid, block, smem, stream>>>(
            static_cast<unsigned int>(args.nelmt), batch, d_basis.ptr, d_JxW.ptr, d_in.ptr, d_out.ptr);
        check(cudaGetLastError(), "kernel launch");
        check(cudaEventRecord(stop, stream), "cudaEventRecord");
        check(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        return 1.0e-3 * static_cast<double>(ms);
    };

    // Warm-up: the GPU has been idle through the host reference and sits
    // at its idle clock, and the first launch loads the module.  Run until
    // bk::ffdrv::warmup_seconds of kernel time have passed, at least twice.
    double warm = 0.0;
    for (int t = 0; t < 2 || warm < bk::ffdrv::warmup_seconds; ++t) {
        warm += run_once();
    }

    // Minimum time over ntests runs.
    double elapsed = 1e300;
    for (int t = 0; t < args.ntests; ++t) {
        elapsed = std::min(elapsed, run_once());
    }

    std::vector<double> out(data.in.size());
    check(cudaMemcpy(out.data(), d_out.ptr, out.size() * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy to host");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return bk::ffdrv::report("float-float", elapsed, out, data.JxW.size(), data.ref, args.tol);
}

} // namespace

int main(int argc, char** argv)
try {
    const bk::ffdrv::Args args = bk::ffdrv::parse_args(argc, argv);
    const bool ok = bk::ffdrv::dispatch(args.p, [&](auto nq_tag) {
        return run_test<decltype(nq_tag)::value>(args);
    });
    return ok ? 0 : 1;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
