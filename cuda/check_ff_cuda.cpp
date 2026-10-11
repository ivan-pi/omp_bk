// check_ff_cuda.cpp -- runs MassOperatorFF of bk1_ff_cuda.cuh on the host
// through cuda_shim.h (one thread per block, one block) and checks it
// against the double reference of bk1_sumfact.h, for nq = 3..10, constant
// and random data, and batches of 1, 3 and 8 elements with a short last
// batch.  Prints the relative L2 errors and ends with ok or FAIL (exit
// status 1).  It checks the kernel's indexing, phases and batching, not
// its execution by many threads on a GPU.
// From the repository root:  make check_ff_cuda && ./check_ff_cuda

#include "cuda_shim.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

// the kernel's dynamic shared memory
namespace bk {
float shared[1 << 20];
}

#include "bk1_ff_cuda.cuh"
#include "bk_common.h"
#include "bk1_sumfact.h"

namespace {

constexpr double tol = 1e-10;

// ||out - ref|| / ||ref|| of the kernel against the double reference, or
// -1 when the batch does not fit the shared-memory stand-in.
template <int nq>
double relative_error(const std::size_t nelmt, const unsigned batch, const bool random_data)
{
    constexpr int nm = nq - 1;
    const std::array<double, nm * nq> b = bk::make_test_basis<double, nm, nq>();
    const std::vector<double> basis(b.begin(), b.end());
    std::vector<double> JxW(nelmt * nq * nq * nq, 1.0);
    std::vector<double> in(nelmt * nm * nm * nm, 3.0);
    if (random_data) {
        bk::fill_random_test_data(in, JxW);
    }
    std::vector<double> ref(in.size());
    std::vector<double> out(in.size(), -1.0);
    bk::SumFactorization<double, nq>(nelmt, basis.data(), JxW.data(), in.data(), ref.data());
    if (bk::mass_operator_ff_traits<nq>::shared_bytes(static_cast<int>(batch)) > sizeof(bk::shared)) {
        return -1.0;
    }
    bk::MassOperatorFF<nq>(static_cast<unsigned>(nelmt), batch,
                           basis.data(), JxW.data(), in.data(), out.data());
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = out[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num) / std::sqrt(den);
}

} // namespace

int main()
{
    threadIdx.x = 0;
    blockIdx.x = 0;
    blockDim.x = 1;
    gridDim.x = 1;

    bool ok = true;
    const unsigned batches[] = {1u, 3u, 8u};
    for (const bool random_data : {false, true}) {
        for (const unsigned batch : batches) {
            // element counts that leave a short last batch for 3 and 8
            const double e[] = {
                relative_error<3>(101, batch, random_data),
                relative_error<4>(101, batch, random_data),
                relative_error<5>(50, batch, random_data),
                relative_error<6>(50, batch, random_data),
                relative_error<7>(25, batch, random_data),
                relative_error<8>(17, batch, random_data),
                relative_error<9>(10, batch, random_data),
                relative_error<10>(9, batch, random_data),
            };
            std::printf("%s data, batch %u:", random_data ? "random  " : "constant", batch);
            for (int i = 0; i < 8; ++i) {
                std::printf(" nq=%d %.2e", i + 3, e[i]);
                ok = ok && (e[i] >= 0.0) && (e[i] < tol);
            }
            std::printf("\n");
        }
    }
    std::printf("%s: MassOperatorFF on the host against double (tol = %g)\n", ok ? "ok" : "FAIL", tol);
    return ok ? 0 : 1;
}
