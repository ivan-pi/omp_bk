#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cstddef>

#include "bk_common.h"
#include "bk5_kernel.h"

using namespace bk;

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

template <typename T, int nq>
void run_test(const std::size_t nelmt, const int ntests)
{
    // Allocation of arrays
    const std::array<T, nq * nq> dbasis = make_test_basis<T, nq, nq>();
    const std::vector<T> G (nelmt * 6 * nq * nq * nq, T(2.0));
    const std::vector<T> in(nelmt * nq * nq * nq, T(3.0));
    std::vector<T>       out(nelmt * nq * nq * nq, T(0.0));

    const std::size_t size_dbasis = dbasis.size();
    const std::size_t size_G      = G.size();
    const std::size_t size_inout  = in.size();

    const T* d_dbasis = dbasis.data();
    const T* d_G      = G.data();
    const T* d_in     = in.data();
    T*       d_out    = out.data();

    // minimum wall time over ntests repetitions
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    double elapsed = std::numeric_limits<double>::max();

    #pragma omp target data \
        map(to: d_dbasis[:size_dbasis]) \
        map(to: d_G[:size_G], d_in[:size_inout]) \
        map(tofrom: d_out[:size_inout])
    for (int t = 0; t < ntests; ++t) {
        auto start = high_resolution_clock::now();

        bk5::SumFactorization<T, nq>(nelmt, d_dbasis, d_G, d_in, d_out);

        auto stop = high_resolution_clock::now();
        duration<double> rep_time = stop - start;
        elapsed = std::min(elapsed, rep_time.count());
    }

    // Performance in GDoF/s
    const auto dof_rate = [&](double seconds) {
        return 1.0e-9 * size_inout / seconds;
    };
    // Effective bandwidth in GB/s: read in + write out + read G
    const auto byte_rate = [&](double seconds) {
        return 1.0e-9 * sizeof(T) * (2 * size_inout + size_G) / seconds;
    };

    std::cout << "SumFactorization -> nelmt = " << nelmt
              << " GDoF/s = " << dof_rate(elapsed)
              << " GB/s = "   << byte_rate(elapsed) << "\n";

    std::cout << "norm = " << norm2(out.data(), out.size()) << "\n";
}

// Default element count. Note: the historical literal was `2 << 18`, which
// is 2^19 = 524288 -- NOT 2^18. The same value is kept here, spelled
// unambiguously.
constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288

int main(int argc, char** argv)
{
    const int nq = (argc > 1) ? std::atoi(argv[1]) : 4;
    const std::size_t nelmt =
        (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;

    // Runtime nq -> compile-time nq: one kernel instantiation per supported
    // order (each case label must pair with its own literal).
    // Note: adding more cases can increase the compilation time.
    switch (nq) {
        case 2: run_test<float, 2>(nelmt, ntests); break;
        case 3: run_test<float, 3>(nelmt, ntests); break;
        case 4: run_test<float, 4>(nelmt, ntests); break;
        case 5: run_test<float, 5>(nelmt, ntests); break;
        case 6: run_test<float, 6>(nelmt, ntests); break;
        case 7: run_test<float, 7>(nelmt, ntests); break;
        case 8: run_test<float, 8>(nelmt, ntests); break;
        default:
            std::cerr << "unsupported nq = " << nq << " (supported: 2..8)\n";
            return 1;
    }

    return 0;
}
