// check_ff_cuda.cpp -- runs MassOperatorFF and MassOperatorFF2 of
// bk1_ff_cuda.cuh on the host through cuda_shim.h (one thread per block,
// one block) and checks them against the double reference of
// bk1_sumfact.h, for nq = 3..10, constant and random data, and batches of
// 1, 3 and 8 elements with a short last batch; the two kernels must also
// give identical outputs.  Prints the relative L2 errors and ends with ok
// or FAIL (exit status 1).  It checks the kernels' indexing, phases and
// batching, not their execution by many threads on a GPU.
// From the repository root:  make check_ff_cuda && ./check_ff_cuda

#include "cuda_shim.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

// the kernels' dynamic shared memory
namespace bk {
alignas(16) float shared[1 << 20];
float2 shared2[1 << 19];
}

#include "bk1_ff_cuda.cuh"
#include "bk_common.h"
#include "bk1_sumfact.h"

namespace {

constexpr double tol = 1e-10;

// ||out - ref|| / ||ref|| of both kernels against the double reference
// (first MassOperatorFF, second MassOperatorFF2), or -1 when the batch does
// not fit the shared-memory stand-in; identical is cleared when the two
// kernels' outputs differ in any value.
template <int nq>
std::array<double, 2> relative_error(const std::size_t nelmt, const unsigned batch, const bool random_data,
                                     bool& identical)
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
    std::vector<double> out2(in.size(), -2.0);
    bk::SumFactorization<double, nq>(nelmt, basis.data(), JxW.data(), in.data(), ref.data());
    if (bk::mass_operator_ff_traits<nq>::shared_bytes(static_cast<int>(batch)) > sizeof(bk::shared2)) {
        return {-1.0, -1.0};
    }
    bk::MassOperatorFF<nq>(static_cast<unsigned>(nelmt), batch,
                           basis.data(), JxW.data(), in.data(), out.data());
    bk::MassOperatorFF2<nq>(static_cast<unsigned>(nelmt), batch,
                            basis.data(), JxW.data(), in.data(), out2.data());
    double num = 0.0;
    double num2 = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = out[i] - ref[i];
        const double d2 = out2[i] - ref[i];
        num += d * d;
        num2 += d2 * d2;
        den += ref[i] * ref[i];
        if (out2[i] != out[i]) {
            identical = false;
        }
    }
    return {std::sqrt(num) / std::sqrt(den), std::sqrt(num2) / std::sqrt(den)};
}

} // namespace

int main()
{
    threadIdx.x = 0;
    blockIdx.x = 0;
    blockDim.x = 1;
    gridDim.x = 1;

    bool ok = true;
    bool identical = true;
    const unsigned batches[] = {1u, 3u, 8u};
    for (const bool random_data : {false, true}) {
        for (const unsigned batch : batches) {
            // element counts that leave a short last batch for 3 and 8
            const std::array<double, 2> e[] = {
                relative_error<3>(101, batch, random_data, identical),
                relative_error<4>(101, batch, random_data, identical),
                relative_error<5>(50, batch, random_data, identical),
                relative_error<6>(50, batch, random_data, identical),
                relative_error<7>(25, batch, random_data, identical),
                relative_error<8>(17, batch, random_data, identical),
                relative_error<9>(10, batch, random_data, identical),
                relative_error<10>(9, batch, random_data, identical),
            };
            for (int kernel = 0; kernel < 2; ++kernel) {
                std::printf("%s data, batch %u, %s:", random_data ? "random  " : "constant", batch,
                            kernel == 0 ? "hi/lo " : "float2");
                for (int i = 0; i < 8; ++i) {
                    std::printf(" nq=%d %.2e", i + 3, e[i][kernel]);
                    ok = ok && (e[i][kernel] >= 0.0) && (e[i][kernel] < tol);
                }
                std::printf("\n");
            }
        }
    }
    std::printf("MassOperatorFF2 against MassOperatorFF: %s\n", identical ? "identical" : "DIFFERENT");
    ok = ok && identical;
    std::printf("%s: MassOperatorFF and MassOperatorFF2 on the host against double (tol = %g)\n", ok ? "ok" : "FAIL", tol);
    return ok ? 0 : 1;
}
