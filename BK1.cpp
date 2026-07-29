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

namespace BK1 {
namespace Serial {

template <typename T, int nq, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ basis,
    const T* __restrict__ JxW,
    const T* __restrict__ in,
    T* __restrict__ out)
{
    constexpr int nm = nq - 1;

    using nm_cview = ndview<const T, nm, nm, nm>;
    using nm_view  = ndview<T, nm, nm, nm>;
    using nq_cview = ndview<const T, nq, nq, nq>;

    #pragma omp target \
        map(to: basis[:nm*nq]) \
        map(to: JxW[:nelmt*nq*nq*nq], in[:nelmt*nm*nm*nm]) \
        map(from: out[:nelmt*nm*nm*nm])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {

        // B(mode i, quad point p) == basis[i * nq + p], shared by all directions
        const ndview<const T, nm, nq> B{basis};

    	// Work arrays: two nq^3 boxes; steps address sub-slices of each box.
    	// Every step assigns its full output sub-slice, so no zeroing is needed.
    	T scratch[2 * nq * nq * nq];
    	const ndview<T, nq, nq, nq> wsp0{scratch};
    	const ndview<T, nq, nq, nq> wsp1{scratch + nq * nq * nq};

        const nm_cview e_in {in  + e * nm_cview::size};
        const nm_view  e_out{out + e * nm_view::size};
        const nq_cview e_JxW{JxW + e * nq_cview::size};

        // step-1 : copy in -> wsp0 (nm^3 sub-block of the nq^3 box)
        for (index_t i = 0; i < nm; i++)
            for (index_t j = 0; j < nm; j++)
                for (index_t k = 0; k < nm; k++)
                    wsp0(i, j, k) = e_in(i, j, k);

        // step-2 : direction 0
        for (index_t p = 0; p < nq; p++)
            for (index_t k = 0; k < nm; k++)
                for (index_t j = 0; j < nm; j++) {
                    T tmp = 0;
                    for (index_t i = 0; i < nm; i++)
                        tmp += wsp0(i, j, k) * B(i, p);
                    wsp1(p, j, k) = tmp;
                }

        // step-3 : direction 1
        for (index_t q = 0; q < nq; q++)
            for (index_t p = 0; p < nq; p++)
                for (index_t k = 0; k < nm; k++) {
                    T tmp = 0;
                    for (index_t j = 0; j < nm; j++)
                        tmp += wsp1(p, j, k) * B(j, q);
                    wsp0(q, p, k) = tmp;
                }

        // step-4 : direction 2
        for (index_t r = 0; r < nq; r++)
            for (index_t q = 0; q < nq; q++)
                for (index_t p = 0; p < nq; p++) {
                    T tmp = 0;
                    for (index_t k = 0; k < nm; k++)
                        tmp += wsp0(q, p, k) * B(k, r);
                    wsp1(p, q, r) = tmp;
                }

        // Reverse operations

        // step-5 : multiply with weights and determinant of Jacobi
        for (index_t r = 0; r < nq; r++)
            for (index_t q = 0; q < nq; q++)
                for (index_t p = 0; p < nq; p++)
                    wsp1(p, q, r) *= e_JxW(p, q, r);

        // step-6 : direction 2
        for (index_t k = 0; k < nm; k++)
            for (index_t q = 0; q < nq; q++)
                for (index_t p = 0; p < nq; p++) {
                    T tmp = 0;
                    for (index_t r = 0; r < nq; r++)
                        tmp += wsp1(p, q, r) * B(k, r);
                    wsp0(q, p, k) = tmp;
                }

        // step-7 : direction 1
        for (index_t j = 0; j < nm; j++)
            for (index_t k = 0; k < nm; k++)
                for (index_t p = 0; p < nq; p++) {
                    T tmp = 0;
                    for (index_t q = 0; q < nq; q++)
                        tmp += wsp0(q, p, k) * B(j, q);
                    wsp1(p, j, k) = tmp;
                }

        // step-8 : direction 0
        for (index_t i = 0; i < nm; i++)
            for (index_t j = 0; j < nm; j++)
                for (index_t k = 0; k < nm; k++) {
                    T tmp = 0;
                    for (index_t p = 0; p < nq; p++)
                        tmp += wsp1(p, j, k) * B(i, p);
                    wsp0(i, j, k) = tmp;
                }

        // step-9 : copy wsp0 -> out
        for (index_t i = 0; i < nm; i++)
            for (index_t j = 0; j < nm; j++)
                for (index_t k = 0; k < nm; k++)
                    e_out(i, j, k) = wsp0(i, j, k);
    }
}

} // namespace Serial
} // namespace BK1

// ---------------------------------------------------------------------------
// Test driver
// ---------------------------------------------------------------------------

template <typename T, int nq>
void run_test(const std::size_t nelmt, const int ntests)
{
    constexpr int nm = nq - 1;

    const std::array<T, nm * nq> basis = make_test_basis<T, nq>();
    const std::vector<T> JxW(nelmt * nq * nq * nq, T(1.0));
    const std::vector<T> in (nelmt * nm * nm * nm, T(3.0));
    std::vector<T>       out(nelmt * nm * nm * nm);

    const std::size_t size_inout = in.size();
    const std::size_t size_JxW   = JxW.size();
    const std::size_t size_basis = basis.size();

    const T* d_basis = basis.data();
    const T* d_JxW   = JxW.data();
    const T* d_in    = in.data();
    T*       d_out   = out.data();

    //---------------------------Serial Kernel---------------------------------
    // minimum wall time over ntests repetitions
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    double elapsed = std::numeric_limits<double>::max();

    #pragma omp target data \
        map(to: d_basis[:size_basis]) \
        map(to: d_JxW[:size_JxW], d_in[:size_inout]) \
        map(tofrom: d_out[:size_inout])
    for (int t = 0; t < ntests; ++t) {
        auto start = high_resolution_clock::now();
        BK1::Serial::SumFactorization<T, nq>(nelmt, d_basis, d_JxW,
                                             d_in, d_out);
        auto stop = high_resolution_clock::now();
        duration<double> rep_time = stop - start;
        elapsed = std::min(elapsed, rep_time.count());
    }

    // Performance in GDoF/s
    const auto dof_rate = [&](double seconds) {
        return 1.0e-9 * size_inout / seconds;
    };
    // Effective bandwidth in GB/s: read in + write out + read JxW
    const auto byte_rate = [&](double seconds) {
        return 1.0e-9 * sizeof(T) * (2 * size_inout + size_JxW) / seconds;
    };

    std::cout << "SumFactorization -> nelmt = " << nelmt
              << " GDoF/s = " << dof_rate(elapsed)
              << " GB/s = "   << byte_rate(elapsed) << "\n";

    std::cout << "Serial norm = " << norm2(out.data(), out.size()) << "\n";
}

// Default element count. Note: the historical literal was `2 << 18`, which
// is 2^19 = 524288 -- NOT 2^18. The same value is kept here, spelled
// unambiguously.
constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288

int main(int argc, char** argv)
{
    const int p = (argc > 1) ? std::atoi(argv[1]) : 2;
    const std::size_t nelmt =
        (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;

    // Runtime p -> compile-time nq: one kernel instantiation per supported
    // order, nq = p + 2 (each case label must pair with its literal + 2).
    switch (p) {
        case 1: run_test<double,  3>(nelmt, ntests); break;
        case 2: run_test<double,  4>(nelmt, ntests); break;
        case 3: run_test<double,  5>(nelmt, ntests); break;
        case 4: run_test<double,  6>(nelmt, ntests); break;
        case 5: run_test<double,  7>(nelmt, ntests); break;
        case 6: run_test<double,  8>(nelmt, ntests); break;
        case 7: run_test<double,  9>(nelmt, ntests); break;
        case 8: run_test<double, 10>(nelmt, ntests); break;
        default:
            std::cerr << "unsupported polynomial order p = " << p
                      << " (supported: 1..8)\n";
            return 1;
    }
    return 0;
}
