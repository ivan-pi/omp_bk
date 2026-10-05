// BK5 (collocated Laplacian at quadrature points) with OpenACC offloading.
//
// The element loop carries a plain `acc parallel loop`; the inner loop nests
// are left without directives so that `nvc++ -Minfo=acc` shows how the
// compiler schedules them. Every per-element work array is declared inside
// the loop body and is therefore private to its iteration.
//
// D(i, n) is the 1-D derivative matrix, shared by all three directions;
// G holds six symmetric metric factors per point, interleaved as
// G(i, j, factor, k) within each element.
//
// Environment: BK_RANDOM=1 fills in and G with seeded random data
// (see acc_common.h); the default is the constant data of BK5.cpp.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "acc_common.h"

namespace bk {

template <typename T, int nq, typename index_t = int>
void SumFactorization(
    const std::size_t nelmt,
    const T* __restrict__ dbasis,
    const T* __restrict__ G,
    const T* __restrict__ in,
          T* __restrict__ out)
{
    // The data clauses are no-ops when the arrays are already present
    // (the driver's `acc data` region), and make the kernel usable on its own.
    #pragma acc parallel loop \
        copyin(dbasis[0:nq*nq]) \
        copyin(G[0:nelmt*6*nq*nq*nq], in[0:nelmt*nq*nq*nq]) \
        copyout(out[0:nelmt*nq*nq*nq])
    for (std::size_t e = 0; e < nelmt; ++e) {

        // D(i, n) == dbasis[i * nq + n], constructed inside the compute
        // region so that it holds the device address.
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
                    T qr = T(0);
                    T qs = T(0);
                    T qt = T(0);

                    for (index_t n = 0; n < nq; ++n) {
                        qr += e_in(n, j, k) * D(i, n);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        qs += e_in(i, n, k) * D(j, n);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        qt += e_in(i, j, n) * D(k, n);
                    }

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
                    for (index_t n = 0; n < nq; ++n) {
                        tmp0 += rqr(n, j, k) * D(n, i);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        tmp0 += rqs(i, n, k) * D(n, j);
                    }
                    for (index_t n = 0; n < nq; ++n) {
                        tmp0 += rqt(i, j, n) * D(n, k);
                    }

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
void run_test(const std::size_t nelmt, const int ntests, const bool random)
{
    // Allocation of arrays
    const std::array<T, nq * nq> dbasis = make_test_basis<T, nq, nq>();
    std::vector<T> G (nelmt * 6 * nq * nq * nq, T(2.0));
    std::vector<T> in(nelmt * nq * nq * nq, T(3.0));
    std::vector<T> out(nelmt * nq * nq * nq, T(0.0));
    if (random) {
        fill_random(G, 1u);
        fill_random(in, 2u);
    }

    [[maybe_unused]] const std::size_t size_dbasis = dbasis.size();   // only the data clause uses it
    const std::size_t size_G      = G.size();
    const std::size_t size_inout  = in.size();

    const T* d_dbasis = dbasis.data();
    const T* d_G      = G.data();
    const T* d_in     = in.data();
    T*       d_out    = out.data();

    // minimum wall time over ntests repetitions; the kernel is synchronous
    // (no async clause), so the host clock brackets the device execution
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    double elapsed = std::numeric_limits<double>::max();

    #pragma acc data \
        copyin(d_dbasis[0:size_dbasis]) \
        copyin(d_G[0:size_G], d_in[0:size_inout]) \
        copyout(d_out[0:size_inout])
    {
        for (int t = 0; t < ntests; ++t) {
            auto start = high_resolution_clock::now();
            SumFactorization<T, nq>(nelmt, d_dbasis, d_G, d_in, d_out);
            auto stop = high_resolution_clock::now();
            duration<double> rep_time = stop - start;
            elapsed = std::min(elapsed, rep_time.count());
        }
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

    std::cout << "norm = " << std::setprecision(9)
              << norm2(out.data(), out.size()) << "\n";
}

// Default element count, the same 2^19 as BK5.cpp.
constexpr std::size_t default_nelmt = std::size_t(1) << 19;   // = 524288

int main(int argc, char** argv)
{
    const int nq = (argc > 1) ? std::atoi(argv[1]) : 4;
    const std::size_t nelmt =
        (argc > 2) ? std::size_t(std::atoll(argv[2])) : default_nelmt;
    const int ntests = (argc > 3) ? std::atoi(argv[3]) : 5;
    const bool random = random_data_requested();

    // Runtime nq -> compile-time nq: one kernel instantiation per supported
    // order (each case label must pair with its own literal).
    switch (nq) {
        case 2: run_test<float, 2>(nelmt, ntests, random); break;
        case 3: run_test<float, 3>(nelmt, ntests, random); break;
        case 4: run_test<float, 4>(nelmt, ntests, random); break;
        case 5: run_test<float, 5>(nelmt, ntests, random); break;
        case 6: run_test<float, 6>(nelmt, ntests, random); break;
        case 7: run_test<float, 7>(nelmt, ntests, random); break;
        case 8: run_test<float, 8>(nelmt, ntests, random); break;
        default:
            std::cerr << "unsupported nq = " << nq << " (supported: 2..8)\n";
            return 1;
    }

    return 0;
}
