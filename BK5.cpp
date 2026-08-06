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

// ---------------------------------------------------------------------------
// BK5 kernel: scalar Laplace operator at quadrature points. Collocated, so
// the number of quadrature points per direction is nq = p + 1 for polynomial
// order p, and there are nq^3 = (p+1)^3 DoFs per element.
// D(i, n) is the 1-D derivative matrix, shared by all three directions;
// G holds six symmetric metric factors per point, interleaved as
// G(i, j, factor, k) within each element.
// ---------------------------------------------------------------------------
namespace bk {

template <typename T, int nq, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ dbasis,
    const T* __restrict__ G,
    const T* __restrict__ in,
          T* __restrict__ out)
{
    #pragma omp target \
        map(to: dbasis[:nq*nq]) \
        map(to: G[:nelmt*6*nq*nq*nq], in[:nelmt*nq*nq*nq]) \
        map(from: out[:nelmt*nq*nq*nq])
    #pragma omp teams loop
    for (std::size_t e = 0; e < nelmt; ++e) {

        // D(i, n) == dbasis[i * nq + n].
        // Constructed inside the target region on purpose: out here the view
        // would capture the *host* pointer (firstprivate structs get no
        // pointer translation), and OpenMP forbids statements between
        // `target` and `teams`, so per-iteration construction is the one
        // correct spot. It costs nothing -- it is a pointer copy the
        // compiler hoists.
        const ndview<const T, nq, nq> D{dbasis};

        using nq_cview = ndview<const T, nq, nq, nq>;
        const nq_cview e_in{in + e * nq_cview::size};
        const ndview<T, nq, nq, nq> e_out{out + e * nq_cview::size};
        // factor index interleaved between j and k
        const ndview<const T, nq, nq, 6, nq> e_G{G + e * (6 * nq * nq * nq)};

        // Intermediate vals
        T s_rqr[nq * nq * nq];
        T s_rqs[nq * nq * nq];
        T s_rqt[nq * nq * nq];
        const ndview<T, nq, nq, nq> rqr{s_rqr};
        const ndview<T, nq, nq, nq> rqs{s_rqs};
        const ndview<T, nq, nq, nq> rqt{s_rqt};

        for (index_t i = 0; i < nq; ++i) {
            for (index_t j = 0; j < nq; ++j) {
                for (index_t k = 0; k < nq; ++k) {

                    // Load geometric factors, coalesced access
                    const T Grr = e_G(i, j, 0, k);
                    const T Grs = e_G(i, j, 1, k);
                    const T Grt = e_G(i, j, 2, k);
                    const T Gss = e_G(i, j, 3, k);
                    const T Gst = e_G(i, j, 4, k);
                    const T Gtt = e_G(i, j, 5, k);

                    // Multiply by D
                    T qr = T(0), qs = T(0), qt = T(0);

                    for (index_t n = 0; n < nq; ++n)
                        qr += e_in(n, j, k) * D(i, n);

                    for (index_t n = 0; n < nq; ++n)
                        qs += e_in(i, n, k) * D(j, n);

                    for (index_t n = 0; n < nq; ++n)
                        qt += e_in(i, j, n) * D(k, n);

                    // Apply chain rule
                    rqr(i, j, k) = Grr * qr + Grs * qs + Grt * qt;
                    rqs(i, j, k) = Grs * qr + Gss * qs + Gst * qt;
                    rqt(i, j, k) = Grt * qr + Gst * qs + Gtt * qt;
                }
            }
        }

        for (index_t i = 0; i < nq; ++i) {
            for (index_t j = 0; j < nq; ++j) {
                for (index_t k = 0; k < nq; ++k) {

                    T tmp0 = T(0);
                    for (index_t n = 0; n < nq; ++n)
                        tmp0 += rqr(n, j, k) * D(n, i);

                    for (index_t n = 0; n < nq; ++n)
                        tmp0 += rqs(i, n, k) * D(n, j);

                    for (index_t n = 0; n < nq; ++n)
                        tmp0 += rqt(i, j, n) * D(n, k);

                    e_out(i, j, k) = tmp0;
                }
            }
        }
    }
}

} // namespace bk

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

        SumFactorization<T, nq>(nelmt, d_dbasis, d_G, d_in, d_out);

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
    const int p = (argc > 1) ? std::atoi(argv[1]) : 2;
    const std::size_t nelmt =
        (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;

    // Runtime p -> compile-time nq: this operator is collocated, so
    // nq = p + 1 (each case label must pair with its literal + 1). p = 1
    // (nq = 2, the linear element) is the smallest meaningful order; p = 0
    // would give nq = 1, a single point with no differentiation.
    // Note: adding more cases can increase the compilation time.
    switch (p) {
        case 1: run_test<float, 2>(nelmt, ntests); break;
        case 2: run_test<float, 3>(nelmt, ntests); break;
        case 3: run_test<float, 4>(nelmt, ntests); break;
        case 4: run_test<float, 5>(nelmt, ntests); break;
        case 5: run_test<float, 6>(nelmt, ntests); break;
        case 6: run_test<float, 7>(nelmt, ntests); break;
        case 7: run_test<float, 8>(nelmt, ntests); break;
        case 8: run_test<float, 9>(nelmt, ntests); break;
        default:
            std::cerr << "unsupported polynomial order p = " << p
                      << " (supported: 1..8)\n";
            return 1;
    }

    return 0;
}
